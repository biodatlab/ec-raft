import gradio as gr
import functools
import json
from inference.core import generate_ec

MAX_INTERVENTIONS = 50

# Load trials from JSON file
try:
    with open('trials.json', 'r') as f:
        trials_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    trials_data = []

def add_intervention(num_interventions):
    num_interventions = min(MAX_INTERVENTIONS, num_interventions + 1)
    updates = [num_interventions]
    updates.extend(gr.update(visible=i < num_interventions) for i in range(MAX_INTERVENTIONS))
    return updates

def remove_intervention(index_to_remove, num_interventions, *all_fields):
    new_num_interventions = num_interventions - 1
    
    all_fields = list(all_fields)
    # Shift all subsequent intervention blocks' data up by one.
    for i in range(index_to_remove, new_num_interventions):
        for k in range(4): # 4 fields per intervention
            source_idx = (i + 1) * 4 + k
            dest_idx = i * 4 + k
            all_fields[dest_idx] = all_fields[source_idx]

    # Clear the fields of what is now the last (and newly empty) block.
    if new_num_interventions >= 0:
        last_block_start_index = new_num_interventions * 4
        all_fields[last_block_start_index] = None  # Dropdown to None
        all_fields[last_block_start_index + 1] = ""
        all_fields[last_block_start_index + 2] = ""
        all_fields[last_block_start_index + 3] = ""

    state_update = [new_num_interventions]
    group_updates = [gr.update(visible=(i < new_num_interventions)) for i in range(MAX_INTERVENTIONS)]
    input_updates = [gr.update(value=v) for v in all_fields]
    
    return state_update + group_updates + input_updates

def toggle_formatted_text(is_visible):
    new_visibility = not is_visible
    button_text = "Hide Prompt" if new_visibility else "Show Prompt"
    return new_visibility, gr.update(visible=new_visibility), gr.update(value=button_text)

def fill_trial_data(trial_index):
    trial = trials_data[trial_index]
    
    num_interventions = len(trial.get('interventions', []))
    
    updates = {
        'title': trial.get('title', ''),
        'brief_summary': trial.get('brief_summary', ''),
        'detailed_description': trial.get('detailed_description', ''),
        'num_interventions_state': num_interventions
    }

    all_fields = []
    # First, populate the interventions that we have data for
    for i in range(num_interventions):
        intervention = trial['interventions'][i]
        updates[f'i_type_{i}'] = intervention.get('type')
        updates[f'i_name_{i}'] = intervention.get('name')
        updates[f'i_desc_{i}'] = intervention.get('desc')
        updates[f'i_other_names_{i}'] = intervention.get('other_names')
        all_fields.extend([
            intervention.get('type', None),
            intervention.get('name', ''),
            intervention.get('desc', ''),
            intervention.get('other_names', '')
        ])

    # Then, clear the rest of the intervention fields
    for i in range(num_interventions, MAX_INTERVENTIONS):
        updates[f'i_type_{i}'] = None
        updates[f'i_name_{i}'] = ""
        updates[f'i_desc_{i}'] = ""
        updates[f'i_other_names_{i}'] = ""
        all_fields.extend([None, "", "", ""])

    visibility_updates = [gr.update(visible=i < num_interventions) for i in range(MAX_INTERVENTIONS)]

    # The final return order must match the outputs list in the click event
    # title, brief_summary, detailed_description, num_interventions_state, all intervention inputs..., all intervention groups
    final_updates = [
        updates['title'],
        updates['brief_summary'],
        updates['detailed_description'],
        updates['num_interventions_state'],
    ] + all_fields + visibility_updates
    
    return final_updates

def clear_inputs():
    num_interventions = 0
    all_fields = [None, "", "", ""] * MAX_INTERVENTIONS
    visibility_updates = [gr.update(visible=False) for _ in range(MAX_INTERVENTIONS)]

    # The return order must match the outputs list for the clear_button click event
    return [
        "",  # title
        "",  # brief_summary
        "",  # detailed_description
        num_interventions,
    ] + all_fields + visibility_updates + [
        "", # formatted_output
        "", # model_output
    ]

if __name__ == "__main__":
    def generate_model_output(*args):
        title = args[0]
        brief_summary = args[1]
        detailed_description = args[2]
        num_interventions = args[3]
        interventions_data = args[4:]
        
        description = ""

        if brief_summary or detailed_description:
            description += "#Study Description\n"
            if brief_summary:
                description += f"Brief Summary\n{brief_summary}\n\n"
            if detailed_description:
                description += f"Detailed Description\n{detailed_description}\n\n"

        interventions = []
        for i in range(num_interventions):
            base_idx = i * 4
            i_type, i_name, i_desc, i_other_names = interventions_data[base_idx:base_idx+4]
            
            if i_type and i_name:
                interventions.append({
                    'type': i_type, 'name': i_name, 'desc': i_desc, 'other_names': i_other_names
                })

        if interventions:
            description += "#Intervention\n"
            for interv in interventions:
                description += f"- {interv['type']} : {interv['name']}\n"
                if interv['desc']:
                    description += f"\t- {interv['desc'].strip()}\n"
                
                if interv['other_names']:
                    description += f"\t- Other Names :\n"
                    other_names_list = [name.strip() for name in interv['other_names'].split(',')]
                    for name in other_names_list:
                        if name:
                            description += f"\t\t- {name}\n"
        
        if not title.strip() and not description.strip():
            return "", ""

        result = generate_ec(title=title, description=description)
        
        return result["prompt"], result["raw_output"]

    with gr.Blocks() as iface:
        gr.Markdown("# EC-RAFT Automated generation of eligibility criteria via RAFT")
        gr.Markdown("Use the fields below to generate eligibility criteria for your study.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Sample Trials")
                sample_buttons = []
                for i in range(min(10, len(trials_data))):
                    btn = gr.Button(f"Sample {i+1}: {trials_data[i].get('title', '')[:30]}...")
                    sample_buttons.append(btn)

            with gr.Column(scale=3):
                title = gr.Textbox(label="Title of the study")
                num_interventions_state = gr.State(value=0)
                prompt_visible_state = gr.State(value=False)
                brief_summary = gr.Textbox(label="Brief Summary", lines=5)
                detailed_description = gr.Textbox(label="Detailed Description", lines=10)

                intervention_inputs = []
                intervention_groups = []
                remove_buttons = []
                
                intervention_type_choices = [
                    "DEVICE", "PROCEDURE", "BEHAVIORAL", "DIETARY_SUPPLEMENT", 
                    "OTHER", "GENETIC", "RADIATION", "DRUG", "BIOLOGICAL", "DIAGNOSTIC_TEST"
                ]

                for i in range(MAX_INTERVENTIONS):
                    with gr.Group(visible=False) as intervention_block:
                        gr.Markdown("### Intervention")
                        i_type = gr.Dropdown(choices=intervention_type_choices, label="Type", elem_id=f"i_type_{i}")
                        i_name = gr.Textbox(label="Name", elem_id=f"i_name_{i}")
                        i_desc = gr.Textbox(label="Description (Optional)", elem_id=f"i_desc_{i}")
                        i_other_names = gr.Textbox(label="Other Names (Comma-separated, Optional)", elem_id=f"i_other_names_{i}")
                        remove_btn = gr.Button("Remove This Intervention")
                        
                        intervention_inputs.extend([i_type, i_name, i_desc, i_other_names])
                        intervention_groups.append(intervention_block)
                        remove_buttons.append(remove_btn)
                
                add_button = gr.Button("Add Intervention")
                gr.Markdown("---")
                
                with gr.Row():
                    generate_button = gr.Button("Generate Eligibility Criteria", variant="primary")
                    clear_button = gr.Button("Clear")

            with gr.Column(scale=3):
                formatted_output = gr.Textbox(label="Formatted Study Text", lines=20, interactive=False, visible=False)
                model_output = gr.Textbox(label="Model Output", lines=20, interactive=False)
                toggle_button = gr.Button("Show/Hide Prompt")

        all_craft_inputs = [title, brief_summary, detailed_description, num_interventions_state] + intervention_inputs

        # Define outputs for sample buttons
        sample_outputs = [
            title, brief_summary, detailed_description, num_interventions_state
        ] + intervention_inputs + intervention_groups

        for i, btn in enumerate(sample_buttons):
            btn.click(
                fn=functools.partial(fill_trial_data, i),
                inputs=[],
                outputs=sample_outputs
            )

        generate_button.click(
            fn=generate_model_output,
            inputs=all_craft_inputs,
            outputs=[formatted_output, model_output]
        )

        clear_outputs = [
            title, brief_summary, detailed_description, num_interventions_state
        ] + intervention_inputs + intervention_groups + [formatted_output, model_output]

        clear_button.click(
            fn=clear_inputs,
            inputs=[],
            outputs=clear_outputs
        )

        toggle_button.click(
            fn=toggle_formatted_text,
            inputs=[prompt_visible_state],
            outputs=[prompt_visible_state, formatted_output, toggle_button]
        )

        add_outputs = [num_interventions_state] + intervention_groups
        add_button.click(fn=add_intervention, inputs=[num_interventions_state], outputs=add_outputs)

        remove_outputs = [num_interventions_state] + intervention_groups + intervention_inputs
        for i, remove_btn in enumerate(remove_buttons):
            remove_btn.click(
                fn=functools.partial(remove_intervention, i),
                inputs=[num_interventions_state] + intervention_inputs,
                outputs=remove_outputs
            )

    iface.launch(share=True)

