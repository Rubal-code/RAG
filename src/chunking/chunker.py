def create_chunks(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks without cutting words.

    We try to end each chunk at a natural boundary
    such as a space or newline.
    """

    chunks = []

    start = 0

    while start < len(text):

        # Initial end position
        end = start + chunk_size

        # If this is not the final chunk,
        # move backward until we find a natural boundary.
        if end < len(text):

            # Look for the last newline or space
            # inside the chunk.
            newline_pos = text.rfind("\n", start, end)
            space_pos = text.rfind(" ", start, end)

            # Choose the closest natural boundary
            boundary = max(newline_pos, space_pos)

            # Use the boundary only if it is reasonably
            # far into the chunk.
            if boundary > start:
                end = boundary

        # Create the chunk
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        next_start = end - overlap

        # Prevent getting stuck
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks