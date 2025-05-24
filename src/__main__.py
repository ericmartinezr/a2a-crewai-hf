import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentSkill, AgentCard
from agent import CrewAIAgent
from agent_executor import CrewAIAgentExecutor
from common.utils.logger import logger


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=10001)
def main(host, port):
    try:
        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id='image_generator',
            name='Image Generator',
            description=(
                'Generate images'
            ),
            tags=['generate image', 'edit image'],
            examples=['Generate a photorealisitc image of raspberry lemonade']
        )

        agent_card = AgentCard(
            name='Image Generator Agent',
            description=('Generate images'),
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=CrewAIAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CrewAIAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill]
        )

        request_handler = DefaultRequestHandler(
            agent_executor=CrewAIAgentExecutor(),
            task_store=InMemoryTaskStore()
        )

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        import uvicorn
        uvicorn.run(server.build(), host=host, port=port)


    except Exception as e:
        logger.error(f"Error al iniciar servidor {e}")
        exit(1)


if __name__ == '__main__':
    main()

