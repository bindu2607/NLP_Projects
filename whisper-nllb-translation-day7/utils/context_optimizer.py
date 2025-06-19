def optimize_context(text, src_lang, tgt_lang):
    """
    Optimize input text context based on source and target languages.
    Replaces idioms and informal expressions for better translation results.
    """

    if not isinstance(text, str):
        raise ValueError("optimize_context expects a string input")

    if src_lang == "en" and tgt_lang == "zh":
        # Lowercase the input for consistency
        text_lower = text.lower()

        # Idioms and expressions dictionary: English -> Simplified Chinese
        idiom_replacements = {
            "break a leg": "祝你好运",
            "hit the sack": "去睡觉",
            "piece of cake": "小菜一碟",
            "under the weather": "身体不舒服",
            "cost an arm and a leg": "非常昂贵",
            "spill the beans": "泄露秘密",
            "let the cat out of the bag": "说漏嘴",
            "hit the books": "用功读书",
            "kick the bucket": "去世（俚语）",
            "bite the bullet": "咬紧牙关面对困难",
            "burn the midnight oil": "熬夜工作或学习",
            "bend over backwards": "极力帮助",
            "out of the blue": "出乎意料",
            "in hot water": "陷入麻烦",
            "call it a day": "收工",
            "add fuel to the fire": "火上加油",
            "cry over spilled milk": "为无法挽回的事悲伤",
            "once in a blue moon": "千载难逢",
            "hit the nail on the head": "一针见血",
            "jump the gun": "操之过急",
            "pull someone’s leg": "开某人玩笑",
            "pull someone's leg": "开某人玩笑",
            "the ball is in your court": "轮到你行动了",
            "on the fence": "犹豫不决",
            "the last straw": "最后一根稻草",
            "throw in the towel": "认输",
            "when pigs fly": "不可能的事情",
            "kill two birds with one stone": "一箭双雕",
            "cut corners": "偷工减料",
            "go the extra mile": "加倍努力",
            "rain cats and dogs": "下倾盆大雨",
            "hang in there": "坚持住",
            "see eye to eye": "意见一致",
            "salt pickle": "咸菜"
        }

        # Replace idioms in the input text
        for phrase, replacement in idiom_replacements.items():
            if phrase in text_lower:
                text_lower = text_lower.replace(phrase, replacement)

        return text_lower

    # For other language pairs or no optimization
    return text
