

def save_to_markdown(response, file_path):
    """
    Saves the AI response to a Markdown file.
    
    Args:
        response (str): The AI-generated response to save.
        file_path (str): The path where the Markdown file will be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response)
        print(f"Response successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the response: {e}")