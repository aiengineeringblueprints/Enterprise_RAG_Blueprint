"""A final deny-by-default document check for the MCP execution path."""

from .identity import Identity


def permitted_documents(documents, identity: Identity):
    from retriever import combine_tag_bitmasks

    if "admin" in identity.roles:
        return documents
    user_mask = combine_tag_bitmasks(identity.roles)
    allowed = []
    for document in documents:
        mask = document.metadata.get("category_bitmask")
        # Missing/unknown/untagged documents do not silently become public.
        if type(mask) is int and mask > 0 and mask & user_mask == mask:
            allowed.append(document)
    return allowed
