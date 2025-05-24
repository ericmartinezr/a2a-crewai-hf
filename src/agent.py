import os
import requests
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Task
from crewai.process import Process
from crewai.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Any, AsyncIterable, Type
from common.utils.logger import logger

load_dotenv()

env = os.environ['ENV']

class ImageModel(BaseModel):
    name: str
    description: str
    path: str

class ImageToolInput(BaseModel):
    """ Input schema for ImageTool """
    prompt: str = Field(description="User prompt")
    session_id: str = Field(description="Session ID")
    image_name: str = Field(description="Image name generated")

class ImageTool(BaseTool):
    name: str = "Image Tool Generator"
    description: str = "Image generation tool that generates images based on a prompt."
    args_schema: Type[BaseModel] = ImageToolInput

    def _run(self, prompt: str, session_id: str, image_name: str) -> str:
        if not prompt or not image_name:
            raise ValueError('Parameters prompt and image_name cannot be empty')

        save_path = os.environ['SAVE_PATH']
        hf_token = os.environ['HF_TOKEN']

        logger.debug(f"ImageTool prompt {prompt}")
        logger.debug(f"ImageTool session_id {session_id}")
        logger.debug(f"ImageTool image_name {image_name}")

        url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
        headers = {
            "Authorization": f"Bearer {hf_token}"
        }
        payload = {
            "inputs": prompt
        }

        try:
            image_path = f"{save_path}/{image_name}.png"
            # Prevents to run out of credits while testing
            if env == "prod":
                response = requests.post(url, headers=headers, json=payload)
                logger.debug(f"Response {response}")
                if response.status_code == 200:
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
            return image_path
        except Exception as e:
            logger.error(f"Error generating image {e}", exc_info=True)
            return -999999999

class CrewAIAgent:

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        
        self.model = LLM(
            model="ollama/gemma3:4b",
            base_url="http://127.0.0.1:11434",
            temperature=0.7,
            stream=True
        )

        #  Crew Agent
        self.image_creator_agent = Agent(
            role='Image Creation Expert',
            goal=(
                "Generate an image using an API based on the user's text prompt."
                ' If the prompt is vague, ask clarifying questions (though the tool currently'
                " doesn't support back-and-forth within one run). Focus on"
                " interpreting the user's request and using the Image Generator"
                ' tool effectively.'
            ),
            backstory=(
                'You are a digital artist powered by AI. You specialize in taking'
                ' textual descriptions and transforming them into visual'
                ' representations using a powerful image generation tool. You aim'
                ' for accuracy and creativity based on the prompt provided.'
            ),
            verbose=True,
            allow_delegation=False,
            tools=[ImageTool()],
            llm=self.model,
            max_iter=3
        )

        # Agent Task
        self.image_creation_task = Task(
            description=(
                "Receive a user prompt: '{user_prompt}'.\nAnalyze the prompt and"
                ' identify if you need to create a new image or edit an existing'
                ' one. Look for pronouns like this, that etc in the prompt, they'
                ' might provide context, rewrite the prompt to include the'
                ' context.If creating a new image, ignore any images provided as'
                " input context.Use the 'Image Generator' tool to for your image"
                ' creation or modification. The tool will expect a prompt which is'
                ' the {user_prompt} and the {session_id}.'
                ' You\'ll also need to generate a fun name for this image and pass it as a paratemer to the tool.' 
                ' This name will be used to save the image.'
            ),
            expected_output='A concise name for the image, a brief description of what\'s going on in the image and the path where the image was saved',
            agent=self.image_creator_agent,
            output_pydantic=ImageModel
        )

        self.image_crew = Crew(
            agents=[self.image_creator_agent],
            tasks=[self.image_creation_task],
            process=Process.sequential,
            verbose=True
        )

    async def invoke(self, query, session_id) -> str: 
        """ Invokes a CrewAI y returns the response """
        inputs = {
            'user_prompt': query,
            'session_id': session_id
        }
        logger.info(f"invoke: Inputs {inputs}")
        response = self.image_crew.kickoff(inputs)
        logger.debug(f"invoke: Response {response}")
        return response
    
    async def stream(self, query: str) -> AsyncIterable[dict[str, Any]]:
        """ Streaming is not supported by CrewAI """
        raise NotImplementedError("Streaming is not supported by CrewAI.")
