from nrclex import NRCLex

def get_emotional_keywords_v2(target_emotions=None):
    """
    Get emotional keywords from NRCLex lexicon.
    
    Args:
        target_emotions (list): List of emotions to target (e.g., ['joy', 'surprise'])
        
    Returns:
        list: Words associated with the target emotions from NRCLex lexicon
    """
    if target_emotions is None:
        target_emotions = ['joy', 'surprise', 'anticipation', 'trust']
    
    # Create an empty NRCLex object to access the lexicon
    emotion_analyzer = NRCLex('')
    
    # Get all words from the lexicon
    emotional_words = []
    
    # Iterate through the affect dictionary to find words with target emotions
    for word, emotions in emotion_analyzer.affect_dict.items():
        # Check if the word has any of our target emotions
        if any(emotion in target_emotions for emotion in emotions):
            emotional_words.append(word)
    
    return emotional_words

text = ['joy', 'surprise', 'anticipation', 'trust']

# Iterate through list
for i in range(len(text)):

    # Create object
    emotion = NRCLex(text[i])

    # Classify emotion
    print('\n\n', text[i], ': ', emotion.top_emotions)

print(get_emotional_keywords_v2())