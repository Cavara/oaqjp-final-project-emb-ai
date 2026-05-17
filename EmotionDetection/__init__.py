# EmotionDetection/__init__.py

import requests
import json

def emotion_detector(text_to_analyse):
    """
    Detects emotions in the given text using IBM Watson NLP Emotion Predict.
    
    Args:
        text_to_analyse (str): Text to analyze for emotions.
    
    Returns:
        dict: Emotion scores and dominant emotion
    """
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }
    
    input_json = {
        "raw_document": {
            "text": text_to_analyse
        }
    }
    
    response = requests.post(url, json=input_json, headers=headers)
    
    try:
        response_dict = json.loads(response.text)
        emotions = response_dict["emotionPredictions"][0]["emotion"]
        
        anger = emotions.get("anger", 0.0)
        disgust = emotions.get("disgust", 0.0)
        fear = emotions.get("fear", 0.0)
        joy = emotions.get("joy", 0.0)
        sadness = emotions.get("sadness", 0.0)
        
        emotion_scores = {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness
        }
        
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        
        return {
            'anger': anger,
            'disgust': disgust,
            'fear': fear,
            'joy': joy,
            'sadness': sadness,
            'dominant_emotion': dominant_emotion
        }
        
    except Exception as e:
        return {"error": f"Failed to analyze emotions: {str(e)}"}