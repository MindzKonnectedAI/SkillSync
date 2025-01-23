from langchain_community.document_loaders import UnstructuredMarkdownLoader

def load_markdown(outputFile):
    markdown_path = outputFile
    # print("markdown_path", markdown_path)
    loader = UnstructuredMarkdownLoader(markdown_path, encoding="utf-8")
    documents = loader.load()
    # print("UnstructuredMarkdownLoaderdocuments", documents)
    # print(f"length of UnstructuredMarkdownLoader documents loaded: {len(documents)}")

    texts = [d.page_content for d in documents]

    # print(f"ltexts: ", texts[0])
    return texts[0]