from agent import build_agent
from ingest import load_document


def main(file_path):

    document = load_document(file_path)
    agent = build_agent(document)

    while True:
        question = input("Enter your query: ")
        if question == "exit":
            break
        print(agent(question))


if __name__ == "__main__":
    main("/Users/filippobuoncompagni/document_explainer/FILIPPO_EPSOP.pdf")
