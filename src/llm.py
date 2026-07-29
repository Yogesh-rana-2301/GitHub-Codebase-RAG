import os

import google.generativeai as genai


class GeminiLLM:
    """
    Thin wrapper around the Google Gemini API (gemini-1.5-flash).

    Why Gemini?
      - Free tier is generous (15 requests/minute, 1 million tokens/day).
      - gemini-1.5-flash has a 1-million-token context window — easily handles
        multiple long chunks in one prompt.
      - The SDK (`google-generativeai`) is simple: configure once, call generate.

    Grounded prompting strategy:
      The model is explicitly told to answer ONLY from the provided context and
      to cite the source file.  If the answer is not in the context, it must say
      so.  This prevents hallucination and makes the system honest about gaps —
      a key production concern interviewers care about.
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: str = None):
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "Gemini API key not found.  "
                "Set the GEMINI_API_KEY environment variable or pass api_key=."
            )
        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(model_name)
        print(f"Gemini model ready: {model_name}")

    def answer(
        self,
        question:       str,
        context_chunks: list[str],
        source_files:   list[str],
    ) -> str:
        """
        Generate an answer grounded in the retrieved context chunks.

        Each chunk is labelled with its source file so the model can cite it.
        The prompt instructs the model to stay within the provided context.
        """
        # Build the context block — label each chunk with its source
        context_block = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}"
            for chunk, src in zip(context_chunks, source_files)
        )

        prompt = f"""You are a technical assistant that answers questions about a GitHub repository.

RULES:
1. Answer using ONLY the information in the CONTEXT below.
2. Always cite which [Source: filename] your answer comes from.
3. If the answer is not present in the CONTEXT, respond with:
   "I could not find this in the provided documentation."
4. Do not use any outside knowledge or make assumptions.

CONTEXT:
{context_block}

QUESTION: {question}

ANSWER (with source citation):"""

        response = self.model.generate_content(prompt)
        return response.text.strip()
