import json
import re

import ollama
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status, get_customer_risk_profile, search_knowledge_base
from .models import Conversation, Message, AgentLog
from .event_queue import DONE, publish


# Ollama local model
ollama_model = settings.OLLAMA_MODEL


# SUPPORT SYSTEM PROMPT --> Maya's job description
SUPPORT_SYSTEM_PROMPT = """
You are Maya, a customer support agent at Nexora AC.
You help customers with issues related to their AC orders.

Your responsibilities:
- Always use your tools to gather facts before responding
- Check order details when customer mentions their order
- Check refund history before making any refund decisions
- Be empathetic but honest

Your personality:
- Friendly and professional
- Patient even when customer is angry
- Clear and concise in your replies
- No emojies

Important rules:
- Always check order details first with the get_order_details tool before responding
- Always call the get_order_details tool at the start of every conversation turn. Never answer directly without tool data.
- Never approve or deny a refund yourself
- If the customer asks for a refund, wants their money back, or complains the product is faulty or not working, you MUST call the escalate_to_manager tool. The tool returns the manager's final decision — then relay that decision to the customer.
- If the customer asks about policies, warranty or refund eligibility, always call search_knowledge_base first
- Never use bold text, bullet points or any markdown formatting. Plain text only.
- Keep replies concise and conversational. Maximum 3-4 sentences. No long paragraphs.
"""


MANAGER_SYSTEM_PROMPT = """
You are a senior support manager at Nexora AC.
A support agent has escalated a customer case to you for a refund decision.

Your responsibilities:
- Review the case summary carefully
- Consider the customer's refund history
- Make a fair and final refund decision
- Give a clear reason for your decision

Your decision options:
- Approve refund — if the case is genuine and within policy
- Deny refund — if the case is suspicious or outside policy
- Escalate to risk team — if you suspect fraud

Important rules:
- Be fair but firm
- Base decision on facts — not emotions
- Always give a specific reason for the decision
- If you suspect fraud or need more data, you MUST call the assess_fraud_risk tool before deciding
- End with a clear final decision line that starts with 'Decision:' (Approve/Deny the refund)
- Keep your response concise and professional
"""


RISK_SYSTEM_PROMPT = """
You are a fraud risk analyst at Nexora AC.
A support manager has sent you a customer profile for risk assessment.

Your job:
- Analyse the customer's order and refund patterns
- Identify suspicious behaviour
- Return a clear risk verdict

Risk levels:
- LOW — genuine customer, normal behaviour
- MEDIUM — some suspicious signals, proceed with caution
- HIGH — clear fraud pattern, recommend denial

Your response format:
- Risk Level: LOW / MEDIUM / HIGH
- Key Signals: what you found suspicious or genuine
- Recommendation: what manager should do

Important rules:
- You MUST call the get_customer_risk_profile tool before giving a verdict
- After the tool returns the profile data, STOP calling tools and immediately give your verdict
- Never call the same tool twice — the profile is already in your context
- Be objective — base verdict on data only
- One bad refund does not make someone fraudulent
- Look for patterns — not isolated incidents
"""

# SUPPORT TOOLS --> Tool schemas, that ai agents will read
SUPPORT_TOOLS = [
    {
        "name": "get_order_details",
        "description": "Fetch complete order details including status, carrier, tracking number and days since order was placed. Use this when customer mentions their order or complains about delivery.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "The order ID to look up"
                }
            },
            "required": ["order_id"]
        }
    },

    {
        "name": "get_refund_history",
        "description": "Get complete refund history for a user. Use this before making any refund related decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID to check refund history for"
                }
            },
            "required": ["user_id"]
        }
    },

    {
        "name": "check_delivery_status",
        "description": "Check current delivery status using tracking number and carrier. Use this when customer complains about delayed or missing delivery.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tracking_number": {
                    "type": "string",
                    "description": "The shipment tracking number"
                },
                "carrier": {
                    "type": "string",
                    "description": "The carrier name for example BlueDart or Delhivery"
                }
            },
            "required": ["tracking_number", "carrier"]
        }
    },

    {
        "name": "escalate_to_manager",
        "description": "Escalate the case to the manager agent for a refund decision. You MUST call this tool whenever the customer requests a refund, asks for their money back, wants a return or complains about a defective/faulty product. It triggers the manager agent, which may consult the risk agent and returns the final decision. Always include customer's user_id in the case summary so the manager can assess fraud risk. Format the case_summary as: 'Customer User ID: X' on the first line, then the order details, refund history and complaint.",
        "input_schema": {
            "type": "object",
            "properties": {
                "case_summary": {
                    "type": "string",
                    "description": "Case summary starting with 'Customer User ID: X' on the first line, then order details, refund history and the customer complaint."
                }
            },
            "required": ["case_summary"]
        }
    },

    {
        "name": "search_knowledge_base",
        "description": "Search Nexora AC company documents including refund policy, warranty policy, and product FAQs. Use this when customer asks about company policies, warranty coverage, warranty claims, refund eligibility, or any general product information that requires accurate company documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information from company documents. Be specific — for example 'refund eligibility within 30 days' instead of just 'refund'."
                }
            },
            "required": ["query"]
        }
    }


]


MANAGER_TOOLS = [
    {
        "name": "assess_fraud_risk",
        "description": "Consult the risk agent to assess fraud risk for a customer. Use this when refund request looks suspicious or customer has multiple refund requests. Pass the user_id to get a risk verdict.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID to assess fraud risk for"
                }
            },
            "required": ["user_id"]
        }
    }
]


RISK_TOOLS = [
    {
        "name": "get_customer_risk_profile",
        "description": "Get complete risk profile for a customer including order history, refund patterns and ratio. Use this to assess fraud risk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID to assess risk for"
                }
            },
            "required": ["user_id"]
        }
    }
]


# convert Anthropic-style tool schemas to Ollama (OpenAI-style) tool schemas
def to_ollama_tools(tools):
    converted = []
    for tool in tools:
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        })
    return converted


# Some Ollama models return tool calls as a JSON array inside the message text
# instead of the structured message.tool_calls field. This extracts them so
# the agent loop always runs the tools the model intended to call.
def extract_tool_calls_from_text(content):
    if not content:
        return None
    candidates = re.findall(r"\[\s*\{[^{}]*\"name\"\s*:\s*\"[^\"]+\"[^{}]*\}\s*\]", content)
    candidates = candidates[:1]
    if not candidates:
        return None
    try:
        parsed = json.loads(candidates[0])
    except json.JSONDecodeError:
        return None
    calls = []
    for item in parsed:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        arguments = item.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        calls.append({
            "name": item["name"],
            "arguments": arguments,
        })
    return calls or None


def get_tool_calls(response):
    if response.message.tool_calls:
        return [
            {"name": tc.function.name, "arguments": tc.function.arguments or {}}
            for tc in response.message.tool_calls
        ]
    return extract_tool_calls_from_text(response.message.content)


# execute_tool() --> bridge between claude and python functions (tools)
def execute_tool(tool_name, tool_input, conversation_id=None):
    try:
        if tool_name == "get_order_details":
            return get_order_details(tool_input["order_id"])
        
        if tool_name == "get_refund_history":
            return get_refund_history(tool_input["user_id"])
        
        if tool_name == "check_delivery_status":
            return check_delivery_status(tool_input["tracking_number"], tool_input["carrier"])
        
        if tool_name == "escalate_to_manager":
            case_summary = tool_input["case_summary"]
            print("escalating to manager=====>", case_summary)
            decision = run_manager_agent(case_summary, conversation_id)
            print("decision===>", decision)
            return decision
        
        if tool_name == 'assess_fraud_risk':
            user_id = tool_input['user_id']
            print("Consulting risk agent for user==>", user_id)
            verdict = run_risk_agent(user_id, conversation_id)
            print("risk verdict==>", verdict)
            return verdict
        
        if tool_name == 'get_customer_risk_profile':
            return get_customer_risk_profile(tool_input['user_id'])
        
        if tool_name == "search_knowledge_base":
            return search_knowledge_base(tool_input["query"])

        return {"error": f"Unknown tool: {tool_name}"}
    except KeyError as e:
        return {"error": f"Tool {tool_name} missing required argument: {e}"}
    except Exception as e:
        return {"error": f"Tool {tool_name} failed: {e}"}


def tool_signature(tool_name, tool_input):
    return (tool_name, json.dumps(tool_input, sort_keys=True))


# Agent Loop --> while loop that loops until the task is done
MAX_AGENT_ITERATIONS = 10


def clean_tool_json_from_content(content):
    if not content:
        return content
    cleaned = re.sub(r'\[\s*\{[^{}]*"name"\s*:\s*"[^"]+"[^{}]*\}\s*\]', "", content)
    return cleaned.strip()


REFUND_REQUEST_PHRASES = [
    "want a refund", "i want a refund", "want refund", "need a refund", "need refund",
    "give me a refund", "give me the refund", "refund my", "refund for", "refund me",
    "money back", "my money back", "return the", "return my", "return this", "i want to return",
    "i want my money", "refund it", "requesting a refund", "please refund",
]
QUESTION_WORDS = ("what", "when", "how", "can", "could", "would", "is there", "are there", "tell me", "explain", "check", "policy", "eligible", "criteria", ".")


def should_auto_escalate(user_message):
    lowered = " " + user_message.lower() + " "
    if any(q in lowered for q in QUESTION_WORDS):
        return False
    return any(p in lowered for p in REFUND_REQUEST_PHRASES)



def run_support_agent(user_message, conversation_id, order_id, user_id):
    conv = Conversation.objects.get(id=conversation_id)

    conversation_messages = []
    for msg in conv.messages.order_by("created_at"):
        conversation_messages.append({
            "role": msg.role,
            "content": msg.content
        })

    escalated = False
    for _ in range(MAX_AGENT_ITERATIONS):
        # send this conversation to local LLM via Ollama
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "system", "content": SUPPORT_SYSTEM_PROMPT + f"\n\nContext: This conversation is about Order #{order_id}, user: {user_id}"}] + conversation_messages,
            tools=to_ollama_tools(SUPPORT_TOOLS),
            options={"num_ctx": 8192, "temperature": 0.3},
        )

        tool_calls = get_tool_calls(response)

        if tool_calls:
            tool_result = []
            for call in tool_calls:
                block_name = call["name"]
                block_input = call["arguments"]

                if block_name == "escalate_to_manager":
                    escalated = True

                event = {"type": "tool_call", "message": f"Calling tool {block_name} with {block_input}"}
                publish(conversation_id, event)
                # log tool call
                AgentLog.objects.create(conversation=conv, event_type="tool_call", message=f"Calling tool {block_name} with {block_input}")

                # execute the tool
                result = execute_tool(block_name, block_input, conversation_id)

                event = {"type": "tool_result", "message": f"{block_name} returned: {str(result)[:200]}"}
                publish(conversation_id, event)

                # log tool result
                AgentLog.objects.create(conversation=conv, event_type="tool_result", message=f"{block_name} returned: {str(result)[:200]}")
                print('executing tool==>', block_name)
                print('block.input===>', block_input)
                tool_result.append({
                    "role": "tool",
                    "name": block_name,
                    "content": str(result)
                })
            
            conversation_messages.append({
                "role": "assistant",
                "content": clean_tool_json_from_content(response.message.content or "")
            })

            conversation_messages.extend(tool_result)

        else:
            final_reply = clean_tool_json_from_content(response.message.content or "")

            # Deterministic escalation fallback: local LLMs sometimes refuse to call
            # escalate_to_manager. If the customer clearly wants a refund and no
            # escalation happened in this run, escalate directly so the manager and
            # risk agents are always involved.
            if not escalated and should_auto_escalate(user_message):
                from .tools import get_order_details, get_refund_history
                order_info = get_order_details(order_id)
                refund_info = get_refund_history(user_id)
                case_summary = (
                    f"Customer User ID: {user_id}\n"
                    f"Order details: {order_info}\n"
                    f"Refund history: {refund_info}\n"
                    f"Customer complaint: {user_message}\n"
                    f"Please make a refund decision."
                )
                print("AUTO-ESCALATING to manager since model did not call the escalate tool")
                decision = run_manager_agent(case_summary, conversation_id)
                escalated = True
                final_reply = ("Decision received from our team: " + decision)

            # Publish final reply
            event = {"type": "final", "message": final_reply}
            publish(conversation_id, event)
            # log final reply
            AgentLog.objects.create(conversation=conv, event_type="final", message=final_reply)

            publish(conversation_id, DONE)
            return final_reply

    fallback = "I'm having trouble processing your request right now. Please try again in a moment."
    publish(conversation_id, {"type": "final", "message": fallback})
    AgentLog.objects.create(conversation=conv, event_type="final", message=fallback)
    publish(conversation_id, DONE)
    return fallback
        

def run_manager_agent(case_summary, conversation_id):
    conv = Conversation.objects.get(id=conversation_id)

    event = {"type": "manager", "message": f"Case received for review: {case_summary[:200]}"}
    publish(conversation_id, event)
    
    AgentLog.objects.create(conversation=conv, event_type="manager", message=f"Case received for review: {case_summary[:200]}")
    
    manager_messages = [
        {"role": "user", "content": case_summary} # user is task giver
    ]

    executed_cache = {}
    for _ in range(MAX_AGENT_ITERATIONS):
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "system", "content": MANAGER_SYSTEM_PROMPT}] + manager_messages,
            tools=to_ollama_tools(MANAGER_TOOLS),
            options={"num_ctx": 8192, "temperature": 0.2},
        )

        tool_calls = get_tool_calls(response)

        if tool_calls:
            tool_result = []
            for call in tool_calls:
                block_name = call["name"]
                block_input = call["arguments"]

                event = {"type": "manager", "message": "Consulting risk agent for fraud assessment..."}
                publish(conversation_id, event)

                # log consulting risk agent
                AgentLog.objects.create(conversation=conv, event_type="manager", message="Consulting risk agent for fraud assessment...")

                signature = tool_signature(block_name, block_input)
                if signature in executed_cache:
                    result = executed_cache[signature]
                    print("manager repeated tool call, using cached verdict")
                else:
                    result = execute_tool(block_name, block_input, conversation_id)
                    executed_cache[signature] = result

                tool_result.append({
                    "role": "tool",
                    "name": block_name,
                    "content": str(result)
                })
            manager_messages.append({
                "role": "assistant",
                "content": clean_tool_json_from_content(response.message.content or "")
            })

            manager_messages.extend(tool_result)
        else:
            decision = clean_tool_json_from_content(response.message.content or "")

            event = {"type": "manager", "message": f"Decision: {decision[:200]}"}
            publish(conversation_id, event)

            AgentLog.objects.create(conversation=conv, event_type="manager", message=f"Decision: {decision[:200]}")
            return decision

    fallback = "Decision: Unable to reach a decision at this time. Recommend manual review of this case."
    event = {"type": "manager", "message": fallback}
    publish(conversation_id, event)
    AgentLog.objects.create(conversation=conv, event_type="manager", message=fallback)
    return fallback


def run_risk_agent(user_id, conversation_id):
    conv = Conversation.objects.get(id=conversation_id)

    event = {"type": "risk", "message": f"Starting fraud assessment for user {user_id}"}
    publish(conversation_id, event)
    
    # log assessment started
    AgentLog.objects.create(conversation=conv, event_type="risk", message=f"Starting fraud assessment for user {user_id}")
    
    risk_messages = [
        {"role": "user", "content": f"Please assess the fraud risk for user ID {user_id}. Use your tool to get their profile and return a verdict."}
    ]

    executed_cache = {}
    for _ in range(MAX_AGENT_ITERATIONS):
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "system", "content": RISK_SYSTEM_PROMPT}] + risk_messages,
            tools=to_ollama_tools(RISK_TOOLS),
            options={"num_ctx": 8192, "temperature": 0.2},
        )

        print("risk tool_calls===>", response.message.tool_calls)

        tool_calls = get_tool_calls(response)
        print("resolved risk tool calls===>", tool_calls)

        if tool_calls:
            tool_result = []
            repeated_call = False
            for call in tool_calls:
                block_name = call["name"]
                block_input = call["arguments"]

                signature = tool_signature(block_name, block_input)
                if signature in executed_cache:
                    result = executed_cache[signature]
                    repeated_call = True
                    print("risk agent repeated tool call, using cached profile")
                else:
                    event = {"type": "risk", "message": f"Calling {block_name} to get customer risk profile..."}
                    publish(conversation_id, event)
                    AgentLog.objects.create(conversation=conv, event_type="risk", message=f"Calling {block_name} to get customer risk profile...")
                    result = execute_tool(block_name, block_input, conversation_id)
                    executed_cache[signature] = result

                tool_result.append({
                    "role": "tool",
                    "name": block_name,
                    "content": str(result)
                })

            if repeated_call:
                # the model already has the profile data - stop calling tools and
                # force it to produce the verdict without any tools available
                force = ollama.chat(
                    model=ollama_model,
                    messages=[
                        {"role": "system", "content": RISK_SYSTEM_PROMPT + "\n\nYou already have the customer risk profile in the conversation above. Do NOT call any tools. Give your verdict now using that data."}
                    ] + risk_messages + [
                        {"role": "assistant", "content": clean_tool_json_from_content(response.message.content or "")}
                    ] + tool_result + [
                        {"role": "user", "content": "Now give your final verdict using the profile data above. Do not call any tools."}
                    ],
                    options={"num_ctx": 8192, "temperature": 0.2},
                )
                verdict = clean_tool_json_from_content(force.message.content or "")
                if not verdict:
                    verdict = "Risk Level: HIGH - Unable to complete automated assessment. Recommend manual review."
                event = {"type": "risk", "message": f"Verdict: {verdict[:200]}"}
                publish(conversation_id, event)
                AgentLog.objects.create(conversation=conv, event_type="risk", message=f"Verdict: {verdict[:200]}")
                return verdict

            risk_messages.append({
                "role": "assistant",
                "content": clean_tool_json_from_content(response.message.content or "")
            })

            risk_messages.extend(tool_result)
        else:
            verdict = clean_tool_json_from_content(response.message.content or "")

            event = {"type": "risk", "message": f"Verdict: {verdict[:200]}"}
            publish(conversation_id, event)
            
            AgentLog.objects.create(conversation=conv, event_type="risk", message=f"Verdict: {verdict[:200]}")
            return verdict

    fallback = "Risk Level: HIGH - Unable to complete automated assessment. Recommend manual review."
    event = {"type": "risk", "message": fallback}
    publish(conversation_id, event)
    AgentLog.objects.create(conversation=conv, event_type="risk", message=fallback)
    return fallback


