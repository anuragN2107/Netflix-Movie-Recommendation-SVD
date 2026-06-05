import gradio as gr
import pickle

# 1. Load the pre-trained SVD model and movie maps
with open("svd_model.pkl", "rb") as f:
    svd_model = pickle.load(f)

with open("movie_dict.pkl", "rb") as f:
    movie_dict = pickle.load(f)

def recommend_movies(user_id, num_recommendations):
    try:
        user_id = int(user_id)
        num_recommendations = int(num_recommendations)
    except ValueError:
        return "⚠️ Error: Please enter valid numerical inputs for User ID and counts."
    
    # 2. Iterate through all valid movies to predict what score this user would give them
    predictions = []
    for movie_id, metadata in movie_dict.items():
        # Predict rating using the saved SVD matrix weights
        prediction = svd_model.predict(user_id, movie_id)
        predictions.append((movie_id, metadata["Name"], metadata["Year"], prediction.est))
    
    # 3. Sort predictions by highest estimated rating
    predictions.sort(key=lambda x: x[3], reverse=True)
    
    # 4. Format top results into a readable string list for the web dashboard
    top_recommendations = predictions[:num_recommendations]
    
    markdown_output = f"### 🎬 Top {num_recommendations} Recommendations for User #{user_id}:\n\n"
    for i, (m_id, name, year, score) in enumerate(top_recommendations, 1):
        markdown_output += f"**{i}. {name}** ({year})  \n*🎯 Predicted Match Rating:* `{score:.2f} / 5.0`\n\n"
        
    return markdown_output

# 5. Build the Gradio Responsive Web UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🍿 Netflix Movie Recommendation Engine (SVD Matrix Factorization)")
    gr.Markdown("Enter a Customer ID to generate personalized, predicted movie matches using collaborative filtering.")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Number(value=712664, label="Target User ID", precision=0)
            count_input = gr.Slider(minimum=5, maximum=20, step=1, value=10, label="Number of Movies to Suggest")
            submit_btn = gr.Button("Generate Recommendations", variant="primary")
            
        with gr.Column(scale=2):
            output_display = gr.Markdown(label="System Output")
            
    # Set button execution trigger logic
    submit_btn.click(
        fn=recommend_movies,
        inputs=[user_input, count_input],
        outputs=output_display
    )
    
    # Add examples directly from your Jupyter Notebook to show off system tracking
    gr.Examples(
        examples=[[712664, 10], [656399, 10], [1331154, 10]],
        inputs=[user_input, count_input]
    )

if __name__ == "__main__":
    demo.launch()