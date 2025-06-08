import json
from client import Client

class BasePromptGenerator:
    def __init__(self, client: Client):
        self.client = client
    
    @staticmethod
    def related_studies_template(nct_id: str, title: str, description: str, criteria: str):
        return f"""Related NCT_ID: {nct_id}
    Related Title: {title}
    Related Description: {description}
    Related Criteria: {criteria}
    """

    def build_studies_context(self, related_studies: list[str]):
        json_related_studies = [json.loads(i) for i in related_studies]
        context = ""
        for i in json_related_studies:
            title = i.get('metadata', {}).get('official_title', "") or i.get('metadata', {}).get('brief_title', "")
            description = i.get('description', "")
            criteria = i.get('criteria', "")
            nct_id = i.get('metadata', {}).get('nct_id', "")
            if title and description:
                context += f"""<STUDY>
    {self.related_studies_template(nct_id, title, description, criteria)}
    </STUDY>"""
        return context

    def extract_study_info(self, study_info: dict, top_n: int = 5):
        metadata = study_info.get('metadata')
        title = metadata.get('official_title', '') or metadata.get('brief_title', '')
        study_id = metadata.get('nct_id')
        description = study_info.get('data')
        desired_criteria = study_info.get('criteria')

        # Ensure we have the minimum required information
        if not title or not description or not desired_criteria or not study_id:
            print(f"Skipping study {study_id}: Missing title or description or desired criteria or study id")
            return None

        query = f'{title} [SEP] {description}'
        relevant_studies = self.client.retrieve_relevant_studies(query, study_id, top_n)
        related_studies_context = self.build_studies_context([i['document'] for i in relevant_studies])
        return related_studies_context, title, description, desired_criteria
