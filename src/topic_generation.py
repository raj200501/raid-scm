from transformers import GPT2LMHeadModel, GPT2Tokenizer

def topic_generation(df):
    """
    Generate topics for the summaries using a pre-trained GPT-2 model.

    Args:
        df (pd.DataFrame): DataFrame containing summaries.

    Returns:
        pd.DataFrame: DataFrame with an added topic column.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    def generate_topic(text):
        inputs = tokenizer.encode(text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    df['topic'] = df['summary'].apply(generate_topic)
    return df
