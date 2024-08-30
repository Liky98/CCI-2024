class Prompts:
    # allganize/Llama-3-Alpha-Ko-8B-Instruct
    LLAMA_ALPHA = '''당신은 인공지능 어시스턴트입니다. 묻는 말에 친절하고 정확하게 답변하세요.'''

    # yanolja/EEVE-Korean-Instruct-10.8B-v1.0
    EEVE = '''You are a helpful assistant.'''
    
    # Qwen/Qwen2-7B-Instruct
    QWEN2 = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'''
    
    #davidkim205/Ko-Llama-3-8B-Instruct
    LLAMA3_DAVID = '''You are a helpful assistant.'''
    
    KIHOON_CUSTOM ='''주어진 대화를 읽고 자세하게 분석한 다음 논리적인 근거를 생각한 뒤 답변해주세요.'''
    KO_GEMMA = '''당신은 질문에 대해서 자세히 설명하는 AI입니다.'''
    BEOMI = '''친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.'''
    YI_KO = '''친절한 챗봇으로서 상대방의 요청에 최대한 자세하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.'''
    EXAONE = '''You are EXAONE model from LG AI Research, a helpful assistant.'''        
    # 기본 모델(blossom 8b)
    DEFAULT ='''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''
    # no system prompt
    NO = None
    @classmethod
    def get_prompt(cls, model_name):
        model_name = model_name.upper()
        if hasattr(cls, model_name):
            return getattr(cls, model_name)
        else:
            return cls.DEFAULT