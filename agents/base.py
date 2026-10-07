import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from lib import trace
from lib.llm import get_llm, text_of


def run_agent(name, system, tools, task, max_steps=6):
    llm = get_llm().bind_tools(tools)
    by_name = {t.name: t for t in tools}
    msgs = [SystemMessage(content=system), HumanMessage(content=task)]
    for _ in range(max_steps):
        try:
            resp = llm.invoke(msgs)
        except Exception as e:
            # Groq rejects a final answer the model wrongly emitted as a tool call
            # (e.g. a tool named "JSON"); the intended answer is in failed_generation.
            body = getattr(e, "body", None)
            err = body.get("error", body) if isinstance(body, dict) else {}
            gen = err.get("failed_generation") if isinstance(err, dict) else None
            if not gen:
                raise
            try:
                parsed = json.loads(gen)
                return json.dumps(parsed.get("arguments", parsed))
            except ValueError:
                return gen
        msgs.append(resp)
        if not resp.tool_calls:
            return text_of(resp)
        for call in resp.tool_calls:
            try:
                result = by_name[call["name"]].invoke(call["args"])
            except Exception as e:
                result = {"error": str(e)}
            trace.log(name, "tool:" + call["name"], {"args": call["args"], "result": result}, "tool call")
            msgs.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"]))
    return '{"error": "max steps reached"}'