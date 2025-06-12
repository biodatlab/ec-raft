from .base_prompt_generator import BasePromptGenerator

class InferencePromptGenerator(BasePromptGenerator):
    @staticmethod
    def create_messages(input):
        return [
            {"role": "user", "content": input},
        ]
    @staticmethod
    def create_input(related_studies_context: str, title: str, description: str):
        return f"""<RELATED_STUDIES>
{related_studies_context}
</RELATED_STUDIES>

Target Study Title: {title}
Target Study Description: {description}

Task Instruction:
1. Based on the "Target Study Title" and "Target Study Description" of the target study, please create a Eligibility Criteria for the target study.
2. please provide a detailed step-by-step logical thinking steps for the Eligibility Criteria you created before creating the whole criteria.
3. Could use example studies (in the <RELATED_STUDIES> section) if they support your justifications, but ensure the reasoning is well-explained and relevant to the study's context
4. Please provide the Eligibility Criteria in the following format (the item within the square brackets [] are the options that you can choose from):
Your long step by step detailed logical thinking steps here.
<FORMATTED_CRITERIA>
#Eligibility Criteria:
Inclusion Criteria:

* Inclusion Criteria 1
* Inclusion Criteria 2
* Inclusion Criteria 3
* ...

Exclusion Criteria:

* Exclusion Criteria 1
* Exclusion Criteria 2
* Exclusion Criteria 3
* ...

##Sex :
[MALE|FEMALE|ALL]
##Ages : 
- Minimum Age : ... Years
- Maximum Age : ... Years
- Age Group (Child: birth-17, Adult: 18-64, Older Adult: 65+) : [ADULT|CHILD|OLDER ADULT] comma separated

##Accepts Healthy Volunteers:
[YES|NO]
</FORMATTED_CRITERIA>
"""

    @staticmethod
    def format_output(justification: str, criteria: str):
        return f"""{justification.replace("<STEP-BY-STEP-DERIVATION-FROM-TITLE-AND-DESCRIPTION>", "").replace("</STEP-BY-STEP-DERIVATION-FROM-TITLE-AND-DESCRIPTION>", "").strip()}
<FORMATTED_CRITERIA>
{criteria.replace('INCLUSION CRITERIA', 'Inclusion Criteria').replace('EXCLUSION CRITERIA', 'Exclusion Criteria')}
</FORMATTED_CRITERIA>
"""
