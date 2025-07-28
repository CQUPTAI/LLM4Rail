# user_info.py
class UserInfo:
    """
    用户信息类，用于存储和管理用户的相关信息
    """
    def __init__(self, user_id: str, ticket_info: dict):
        """
        :param user_id: 用户ID
        :param ticket_info: 用户的火车票信息
        """
        self.user_id = user_id
        self.ticket_info = ticket_info

    def parse_user_info(self):
        """
        解析用户id
        :return: 用户id所蕴含信息的字典
        """
        return [{'出生日期': '2004/5/18', '性别':'male', '年龄':'adult', '出生地':'Chongqing'},
                {'出生日期': '2004/5/18', '性别': '男', '年龄': '21', '出生地': '重庆市 大渡口区', '当前日期': "2025-06-07"}]
        # return

    def get_user_info(self) -> dict:
        """
        获取用户信息
        :return: 用户的相关信息字典
        """
        return {"user_id": self.user_id, "ticket_info": self.ticket_info}
