from project.components.inferance import Prediction_Pipeline
import matplotlib.pyplot as plt
import numpy as np
 
if __name__ == "__main__":
    model_path = "artifacts/training/best_chest_xray_model.keras"
    test_img = "artifacts/data_ingestion/validation_dataset/images/1314_test_1_.png"

    pipeline = Prediction_Pipeline(model_path=model_path)

    orig, mask, result = pipeline.predict(test_img)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original X-ray")
    plt.imshow(orig)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("AI Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Detection Result")
    plt.imshow(result)
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("AI Detected Region")
    plt.imshow(orig)
    plt.imshow(mask.squeeze(), cmap='Reds', alpha=0.4)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

