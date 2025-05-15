import re


def match_tags(tag: str, msg_content: str, need_split: bool = True):
    r"""Parse search terms from a message.

    Args:
        msg_content (str): The content of the message.

    Returns:
        list: A list of search strategy.
    """
    pattern = r"<{tag}>(.*?)</{tag}>".format(tag=tag)
    matches = re.findall(
        pattern, msg_content, re.DOTALL
    )  #! re.DOTALL: . matches any character, including a newline
    if not need_split:
        return [line.strip() for line in matches if line]
    match_list = []
    for i, content in enumerate(matches):
        match_list.extend(content.split('\n'))
    match_list = [line.strip() for line in match_list if line]

    return match_list
