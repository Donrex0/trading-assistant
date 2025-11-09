from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Optional
import os
from utils.utils import get_model_path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT pre-trained model
    Classifies text into: Positive, Negative, Neutral
    """
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize sentiment analyzer with optional caching
        
        Args:
            cache_dir: Directory to cache model weights
        """
        self.model_name = "yiyanghkust/finbert-tone"
        self.cache_dir = cache_dir
        
        # Initialize model and tokenizer with caching
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=cache_dir
        )
        
        # Map model outputs to sentiment labels
        self.label_map = {
            0: "Neutral",
            1: "Positive",
            2: "Negative"
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            "Positive": 0.7,
            "Negative": 0.7,
            "Neutral": 0.6
        }

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment scores and prediction
        """
        try:
            # Tokenize and analyze
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            
            # Get probabilities for each class
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze()
            probabilities = probabilities.detach().numpy()
            
            # Get predicted class
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            predicted_label = self.label_map[predicted_class]
            confidence = probabilities[predicted_class]
            
            # Apply confidence threshold
            if confidence < self.confidence_thresholds[predicted_label]:
                predicted_label = "Neutral"
                confidence = probabilities[0]
            
            # Create result dictionary
            result = {
                "sentiment": predicted_label,
                "confidence": confidence,
                "scores": {
                    "Neutral": probabilities[0],
                    "Positive": probabilities[1],
                    "Negative": probabilities[2]
                }
            }
            
            # Log high-confidence predictions
            if confidence > 0.9:
                logger.info(f"[✅] High-confidence prediction: {predicted_label} ({confidence:.2f}) - {text[:50]}...")
            
            return result

        except Exception as e:
            logger.error(f"[❌] Error analyzing sentiment: {str(e)}")
            return {
                "sentiment": "Neutral",
                "confidence": 0.0,
                "scores": {
                    "Neutral": 1.0,
                    "Positive": 0.0,
                    "Negative": 0.0
                }
            }

    def analyze_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts with batch processing
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment scores and predictions
        """
        try:
            # Batch processing
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            
            # Get probabilities for each class
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.detach().numpy()
            
            results = []
            for probs in probabilities:
                predicted_class = np.argmax(probs)
                predicted_label = self.label_map[predicted_class]
                confidence = probs[predicted_class]
                
                # Apply confidence threshold
                if confidence < self.confidence_thresholds[predicted_label]:
                    predicted_label = "Neutral"
                    confidence = probs[0]
                
                result = {
                    "sentiment": predicted_label,
                    "confidence": confidence,
                    "scores": {
                        "Neutral": probs[0],
                        "Positive": probs[1],
                        "Negative": probs[2]
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"[❌] Error analyzing batch: {str(e)}")
            return [{
                "sentiment": "Neutral",
                "confidence": 0.0,
                "scores": {
                    "Neutral": 1.0,
                    "Positive": 0.0,
                    "Negative": 0.0
                }
            } for _ in texts]

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batch
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries with sentiment analysis results
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probabilities = probabilities.detach().numpy()
        
        results = []
        for i, probs in enumerate(probabilities):
            predicted_class = probs.argmax()
            result = {
                "sentiment": self.label_map[predicted_class],
                "confidence": probs[predicted_class],
                "scores": {
                    "Neutral": probs[0],
                    "Positive": probs[1],
                    "Negative": probs[2]
                },
                "text": texts[i]
            }
            results.append(result)
        
        return results

    def save_model(self):
        """
        Save the model to the models directory
        """
        model_path = get_model_path("finbert_sentiment")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        print(f"[✅] Model saved to: {model_path}")

    @staticmethod
    def load_model():
        """
        Load the saved model from the models directory
        """
        model_path = get_model_path("finbert_sentiment")
        if os.path.exists(model_path):
            return FinBERTSentimentAnalyzer.from_pretrained(model_path)
        return FinBERTSentimentAnalyzer()

    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load model from a specific path
        """
        instance = cls()
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return instance

if __name__ == "__main__":
    # Example usage
    analyzer = FinBERTSentimentAnalyzer()
    
    # Test with sample texts
    sample_texts = [
        "Bitcoin has shown strong growth potential this quarter.",
        "The recent market crash has caused significant losses.",
        "The market is currently stable with no major movements."
    ]
    
    results = analyzer.analyze_batch(sample_texts)
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("Scores:")
        for label, score in result['scores'].items():
            print(f"  {label}: {score:.2%}")
