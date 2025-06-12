import json
from client import Client
from .base_prompt_generator import BasePromptGenerator

class ChainOfThoughtPromptGenerator(BasePromptGenerator):
    system_prompt = "You are a derive chatbot designed to generate step-by-step derivation that gradually derived the Desired criteria form the Title and Description of a study. Your task is to analyze the title and description of a study and build logical, step-by-step deriviation that connect the study's key elements to the desired criteria. Reference related example studies if they reinforce your justifications. You must assume the desired criteria are correct (as it was already reviewed by specialists) and develop arguments to support them based on the study context and relevant research insights."
    @staticmethod
    def create_messages(input):
        return [
            {"role": "system", "content": ChainOfThoughtPromptGenerator.system_prompt},
            {"role": "user", "content": input},
        ]

    @staticmethod
    def user_prompt_template(related_studies_context: str, title: str, description: str, desired_criteria: str):
        user_prompt_template = """<RELATED_STUDIES>{related_studies_context}</RELATED_STUDIES>

Title: {title}
Description: {description}
Desired criteria: {desired_criteria}

Task Instructions:
1. Derive a step-by-step derivation starting from the "Title" and "Description" provided, gradually building up to support the "Desired criteria".
2. Clearly explain the rationale behind each parameter of all criteria, including values, thresholds, and other specific details.
3. Please use relevant related studies (in the <RELATED_STUDIES> section) if they support your justifications, but ensure the reasoning is well-explained and relevant to the study's context. If the example studies have a conflicting criteria, please explain why criterion in Desired criteria is more relevant to the study.
4. Avoid mentioning that the desired criteria were already provided, and please do not cite the "Desired criteria" directly in your justification.
5. Derive the criteria on the high level first, then derive the specific criteria/values/parameters.
6. You should give the justification/derivation first before giving out any thing about the specific criteria/values/parameters. 
    6.1) Use related studies to justify/derive the parameters of the criteria
    - BAD EXAMPLE: The study requires participants with an ejection fraction of <40%, as this ensures reliable outcomes in the population of interest.
    - GOOD EXAMPLE: To evaluate the efficacy of the intervention on heart failure, it is essential to ensure that participants have significant but stable cardiac impairment. This prevents confounding by acute conditions and ensures reliable outcomes. Studies like NCT03536880 set an ejection fraction threshold of <40% for this reason, reflecting patients with systolic dysfunction while avoiding excessively low values that could result in high mortality unrelated to the intervention.
    6.2) If there is any relevant related studies you should recite relevant information from it before giving out the parameters.
    - BAD EXAMPLE: The study requires a platelet count of >50,000, which is a reasonable threshold to ensure that patients are not at risk of bleeding complications, as seen in NCT00216866.
    - GOOD EXAMPLE: As the study aims to investigate post-thrombotic syndrome, it is important to ensure that patients are not at risk of bleeding complications. As seen in NCT00216866, A platelet count of >50,000 is a reasonable threshold to ensure this.
    6.3) Conflict resolution:
    - BAD EXAMPLE: Study NCT01234567 required patients with an HbA1c < 7.0% to ensure a well-controlled diabetes population. However, our study requires an HbA1c > 8.5% to target poorly controlled diabetes, which aligns with our objectives.
    - GOOD EXAMPLE: The study aims to assess the impact of a novel intervention in patients with poorly controlled diabetes, ensuring that enrolled individuals represent a high-risk population that may benefit from treatment. Unlike NCT01234567, which set an HbA1c cutoff of <7.0% to study glycemic stability in well-controlled patients, our threshold of >8.5% targets individuals who struggle with glycemic control, making them more suitable for evaluating the intervention's effectiveness in a real-world, high-risk setting.
Remember: Derive step by step from the title and description, Rationale before criteria, and Avoid mentioning that the desired criteria were already provided

Response Format:
<STEP-BY-STEP-DERIVATION-FROM-TITLE-AND-DESCRIPTION>
Your long step by step detailed logical derivation here.
</STEP-BY-STEP-DERIVATION-FROM-TITLE-AND-DESCRIPTION>
"""

        return user_prompt_template.format(related_studies_context=related_studies_context, title=title, description=description, desired_criteria=desired_criteria)

    
    def get_messages_for_CoT_huggingface(self, related_studies_context: str, title: str, description: str, desired_criteria: str):
        return self.create_messages(self.user_prompt_template(related_studies_context, title, description, desired_criteria))