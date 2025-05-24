from typing import override
from agent import CrewAIAgent
from a2a.types import (
    Task,
    InvalidParamsError, 
    UnsupportedOperationError
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from common.utils.logger import logger

class CrewAIAgentExecutor(AgentExecutor):
    
    def __init__(self):
        self.agent = CrewAIAgent()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        logger.debug(f"User input: {query}")
        logger.debug(f"Context id {context.context_id}")
        try:
            result = await self.agent.invoke(query, context.context_id)
            event_queue.enqueue_event(new_agent_text_message(result.raw))
            logger.debug(f"Final result {result}")
        except Exception as e:
            logger.error(f"Error invoking the agent {e}", exc_info=True)
            raise ServerError(
                error=ValueError(f"Error invoking agent: {e}")
            ) from e
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
    
    def _validate_request(self, context: RequestContext) -> bool:
        return False