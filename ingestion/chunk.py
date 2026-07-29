from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]

def chunk_file(file_path: Path):
    """
    Splits the content of a Markdown file into chunks based on specified headers.

    Args:
        file_path (Path): The path to the Markdown file to be split.

    Returns:
        List[Chunk]: A list of chunks, each containing metadata and page content.
    """
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    markdown_text = file_path.read_text()
    return splitter.split_text(markdown_text)
