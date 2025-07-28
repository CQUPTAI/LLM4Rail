# utils.py
def format_ticket_info(ticket_info: dict) -> str:
    """
    格式化票务信息
    :param ticket_info: 票务信息字典
    :return: 格式化后的票务信息字符串
    """
    return f"票务状态: {ticket_info['ticket_status']}, 票价: {ticket_info['ticket_price']}"
