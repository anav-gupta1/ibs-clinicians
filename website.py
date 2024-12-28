import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

device = torch.device('cpu')  # Force CPU usage for Streamlit Cloud

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SymptomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4)  # Assuming 4 severity levels
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, criterion, optimizer, device, epochs=50):
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return all_preds, all_labels

def cluster_based_oversampling(X, y, cluster_labels, minority_class=0):
    X_resampled = list(X)
    y_resampled = list(y)
    minority_samples = X[y == minority_class]
    minority_clusters = cluster_labels[y == minority_class]

    for cluster in np.unique(minority_clusters):
        cluster_samples = minority_samples[minority_clusters == cluster]
        num_samples_in_cluster = len(cluster_samples)

        if num_samples_in_cluster < 2:
            X_resampled.extend(cluster_samples)
            y_resampled.extend([minority_class] * num_samples_in_cluster)
            continue

        num_samples_to_generate = num_samples_in_cluster
        for _ in range(num_samples_to_generate):
            sample_1, sample_2 = random.sample(list(cluster_samples), 2)
            synthetic_sample = (sample_1 + sample_2) / 2
            X_resampled.append(synthetic_sample)
            y_resampled.append(minority_class)

    return np.array(X_resampled), np.array(y_resampled)

def analyze_symptom_severity_matrix(df, target_symptom):
    try:
        # Fill missing values
        df = df.fillna(0)
        
        # Prepare features and target
        X = df.drop(columns=[target_symptom])
        y = df[target_symptom]
        feature_names = X.columns.tolist()
        
        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        n_classes = len(label_encoder.classes_)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA with more components (like in CLI version)
        pca = PCA(n_components=1024)  # Match CLI version
        X_pca = pca.fit_transform(X_scaled)
        
        # Add clustering and oversampling
        kmeans = KMeans(n_clusters=10, random_state=42)
        clusters = kmeans.fit_predict(X_pca)
        X_resampled, y_resampled = cluster_based_oversampling(X_pca, y_encoded, clusters)
        
        # Create datasets and dataloaders
        X_tensor = torch.FloatTensor(X_resampled).to(device)
        y_tensor = torch.LongTensor(y_resampled).to(device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
        
        # Initialize model with same architecture
        model = init_model(
            input_dim=X_pca.shape[1],
            hidden_dim=128,
            output_dim=n_classes,
            feature_names=feature_names
        )
        if model is None:
            return None, None, None, None, None, None, None
        
        # Training setup with scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        # Training with early stopping
        best_accuracy = 0
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        # Train model
        for epoch in range(100):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation phase
            if (epoch + 1) % 5 == 0:
                model.eval()
                test_preds = []
                test_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        _, predicted = torch.max(outputs.data, 1)
                        test_preds.extend(predicted.cpu().numpy())
                        test_labels.extend(batch_y.cpu().numpy())
                
                test_acc = accuracy_score(test_labels, test_preds)
                scheduler.step(test_acc)
                
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(all_labels, all_preds),
            'Precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'Recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'F1 Score': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Calculate feature importance using permutation importance
        importance_scores = calculate_permutation_importance(model, X_resampled, y_resampled)
        
        # Create feature importance dictionary
        feature_importance = {}
        for idx, score in enumerate(importance_scores):
            if idx < len(feature_names):
                feature_importance[feature_names[idx]] = float(score)
        
        # Sort features by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
        
        return model, scaler, pca, label_encoder, feature_importance, metrics, conf_matrix

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

# Neural Network Model
class MatrixBasedAcidityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, feature_names=None):
        super().__init__()
        self.feature_names = feature_names

        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.abundance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.interaction_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        self.feature_weights = torch.sigmoid(self.feature_attention(x))
        weighted_input = x * self.feature_weights
        abundance_features = self.abundance_network(weighted_input)
        interaction_features = self.interaction_network(weighted_input)
        combined = torch.cat([abundance_features, interaction_features], dim=1)
        output = self.final_layers(combined)
        return output

    def get_feature_importance(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            _ = self(X_tensor)
            feature_weights = self.feature_weights.cpu().numpy()
            mean_weights = feature_weights.mean(axis=0)
            normalized_weights = mean_weights / mean_weights.sum()
            
            feature_importance = {}
            if self.feature_names is not None:
                for idx, name in enumerate(self.feature_names):
                    if idx < len(normalized_weights):
                        feature_importance[name] = float(normalized_weights[idx])
            
            feature_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True))
            return feature_importance

def calculate_permutation_importance(model, X, y, metric='accuracy', n_repeats=5):
    """
    Calculate feature importance using permutation importance method.
    """
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    
    # Calculate baseline score
    with torch.no_grad():
        baseline_output = model(X_tensor)
        baseline_preds = torch.argmax(baseline_output, dim=1).cpu().numpy()
        baseline_score = accuracy_score(y, baseline_preds)
    
    importance_scores = []
    
    # Calculate importance for each feature
    for feature_idx in range(X.shape[1]):
        feature_scores = []
        
        for _ in range(n_repeats):
            # Create a copy of the data
            X_permuted = X.copy()
            # Shuffle the feature
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Calculate score with permuted feature
            X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
            with torch.no_grad():
                permuted_output = model(X_permuted_tensor)
                permuted_preds = torch.argmax(permuted_output, dim=1).cpu().numpy()
                permuted_score = accuracy_score(y, permuted_preds)
            
            # Importance is decrease in performance
            importance = baseline_score - permuted_score
            feature_scores.append(importance)
        
        # Average importance over repeats
        importance_scores.append(np.mean(feature_scores))
    
    return np.array(importance_scores)

def init_model(input_dim, hidden_dim, output_dim, feature_names):
    try:
        model = MatrixBasedAcidityNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            feature_names=feature_names
        ).to(device)
        return model
    except Exception as e:
        st.error(f"Model initialization error: {str(e)}")
        return None

def main():
    # Set page config and theme
    st.set_page_config(
        page_title="Neural Network Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4A4A4A;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #666666;
        }
        .stSelectbox [data-baseweb="select"] {
            background-color: #2C2C2C;
        }
        .stDataFrame {
            background-color: #2C2C2C;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("Neural Network Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes symptom patterns and their correlations using neural networks. 
    Upload your dataset to begin the analysis.
    """)

    # File uploader with debug information
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Read the CSV file using Streamlit's cache
            @st.cache_data
            def load_csv():
                return pd.read_csv(uploaded_file)
            
            # Load and display data preview
            df = load_csv()
            st.write("Data Preview:")
            st.write(df.head())
            
            # Symptom name mapping
            symptom_display_names = {
                "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Acidity]": "Acidity Frequency",
                "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Bloating]": "Bloating Frequency",
                "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Flatulence/Gas/Fart]": "Flatulence Frequency",
                "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Constipation]": "Constipation Frequency",
                "How many days in a week do you generally experience the following symptoms? (Please respond for all symptoms) [Burping]": "Burping Frequency",
                "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Bloating]": "Bloating Severity",
                "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Acidity/Burning]": "Acidity Severity",
                "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Constipation]": "Constipation Severity",
                "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Loose Motion/Diarrhea]": "Diarrhea Severity",
                "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Flatulence/Gas/Fart]": "Flatulence Severity",
                "How much does these symptoms bother your daily life from 1-10?  (Please respond for all symptoms) [Burping]": "Burping Severity"
            }

            # Get symptom columns and update dropdown
            symptom_columns = [col for col in df.columns if col in symptom_display_names]
            
            if not symptom_columns:
                st.warning("No symptom columns found.")
                return

            # Update the symptom selection dropdown to use the new display names
            selected_symptom_display = st.sidebar.selectbox(
                "Select Symptom to Analyze",
                [symptom_display_names[col] for col in symptom_columns]
            )

            # Convert display name back to original column name for processing
            selected_symptom = {v: k for k, v in symptom_display_names.items()}[selected_symptom_display]

            # Main analysis
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Analysis Progress")
                
                # Run analysis
                model, scaler, pca, label_encoder, feature_importance, metrics, conf_matrix = \
                    analyze_symptom_severity_matrix(df, selected_symptom)

                # Filter out symptom columns from feature importance
                filtered_features = {k: v for k, v in feature_importance.items() 
                                    if k not in symptom_columns}

                # Now create the table with filtered features
                st.subheader("Top 10 Most Important Features")
                top_features = dict(list(filtered_features.items())[:10])
                importance_df = pd.DataFrame({
                    'Feature': list(top_features.keys()),
                    'Importance Score': list(top_features.values())
                }).round(4)

                # Style and display the dataframe
                st.dataframe(
                    importance_df.style.background_gradient(cmap='viridis', subset=['Importance Score'])
                    .format({'Importance Score': '{:.4f}'}),
                    use_container_width=True
                )

                # Display metrics
                st.subheader("Model Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': metrics.keys(),
                    'Value': [f"{v:.4f}" for v in metrics.values()]
                })
                st.dataframe(metrics_df, use_container_width=True)

            with col2:
                # Confusion Matrix display
                st.subheader("Confusion Matrix")
                fig = px.imshow(
                    conf_matrix,
                    labels=dict(x="Predicted", y="Actual"),
                    x=label_encoder.classes_,
                    y=label_encoder.classes_,
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white'
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Error details:", str(e))

if __name__ == "__main__":
    main()