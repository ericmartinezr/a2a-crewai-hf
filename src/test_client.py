from a2a.client import A2AClient
import httpx
from uuid import uuid4
from a2a.types import (
    Message,
    TextPart,
    SendStreamingMessageRequest,
    MessageSendParams,
    MessageSendConfiguration
)
from common.utils.logger import logger
from pydantic_core import from_json, ValidationError

async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        client = await A2AClient.get_client_from_agent_card_url(
            httpx_client, 'http://localhost:9999'
        )

        prompt = 'Generate an image of a cute cat.'

        message = Message(
            role='user',
            parts=[TextPart(text=prompt)],
            messageId=str(uuid4()),
        )

        payload = MessageSendParams(
            id=str(uuid4()),
            message=message,
            configuration=MessageSendConfiguration(
                acceptedOutputModes=['text']
            )
        )

        try:
            stream_response = client.send_message_streaming(
                SendStreamingMessageRequest(
                    id=str(uuid4()),
                    params=payload
                )
            )
            async for chunk in stream_response:
                image_model = chunk.model_dump_json()
                image_model_json = from_json(image_model)
                image_model_json_result = from_json(image_model_json['result']['parts'][0]['text'].replace("```json", "").replace("```", ""))
                
                logger.debug(f"ImageModelJson {image_model_json}")
                logger.debug(f"ImageModel {image_model}")
                logger.debug(f"Json Result {image_model_json_result}")
                
                print(f"The response contains the following image data")
                print(f"""
                    Name: {image_model_json_result['name']}
                    Description: {image_model_json_result['description']}
                    Path: {image_model_json_result['path']}
                """)
        except ValueError as e:
            logger.error(f"Failed at deserializing ImageModel model")
            logger.error(e, exc_info=True)
        except ValidationError as e:
            logger.error(f"Returned value isnt of type ImageModel")
            logger.error(e, exc_info=True)
        except Exception as e:
            logger.error(f"Failed to complete the call to the agent")
            logger.error(e, exc_info=True)
            print("Failed to complete the call", e)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
