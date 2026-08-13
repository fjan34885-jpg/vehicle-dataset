
# YOLOv8 Vehicle Detection Streamlit App

This is a Streamlit application for detecting vehicles in images and videos using a custom-trained YOLOv8 model.

## 🚀 How to Deploy on Streamlit Cloud

1.  **Fork this Repository**: Make sure all these files (`app.py`, `requirements.txt`, `best.pt`, `README.md`) are in the root of your GitHub repository.
2.  **Go to Streamlit Cloud**: Sign up or log in at [share.streamlit.io](https://share.streamlit.io/).
3.  **Deploy an app**: Click on 'New app' or 'Deploy an app'.
4.  **Connect to GitHub**: Select your GitHub repository where you pushed these files.
5.  **Configure Deployment**:
    *   **Repository**: Select your repository.
    *   **Branch**: Choose the branch (e.g., `main`).
    *   **Main file path**: Set this to `app.py`.
    *   **Python version**: Use a compatible Python version (e.g., `3.9`, `3.10`).
6.  **Deploy!**: Click the 'Deploy!' button. Streamlit Cloud will install the dependencies and deploy your app.

## ⚙️ Local Development (Optional)

To run this application locally:

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd Deployment
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .venv\Scriptsctivate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

Your application will open in your web browser.

## Files in this Directory

*   `app.py`: The main Streamlit application script.
*   `requirements.txt`: Python package dependencies.
*   `best.pt`: The trained YOLOv8 model weights.
*   `README.md`: This file, providing deployment instructions.
