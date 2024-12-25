document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const resultsSection = document.getElementById('results');
    const mainPrediction = document.getElementById('main-prediction');
    const confidence = document.getElementById('confidence');
    const probabilities = document.getElementById('probabilities');

    // Preview image before upload
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const imageFile = imageInput.files[0];
        
        if (!imageFile) {
            alert('Veuillez sélectionner une image');
            return;
        }

        formData.append('image', imageFile);

        try {
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.error) {
                alert('Erreur: ' + result.error);
                return;
            }

            // Display results
            resultsSection.style.display = 'block';
            mainPrediction.textContent = result.class_name.charAt(0).toUpperCase() + result.class_name.slice(1);
            confidence.textContent = (result.confidence * 100).toFixed(2);

            // Display all probabilities
            probabilities.innerHTML = '';
            result.all_probabilities.forEach(prob => {
                const probDiv = document.createElement('div');
                probDiv.className = 'mb-2';
                
                const label = document.createElement('div');
                label.textContent = `${prob.class.charAt(0).toUpperCase() + prob.class.slice(1)}: ${(prob.probability * 100).toFixed(2)}%`;
                
                const progressContainer = document.createElement('div');
                progressContainer.className = 'probability-bar';
                
                const progressBar = document.createElement('div');
                progressBar.className = 'probability-fill';
                progressBar.style.width = `${prob.probability * 100}%`;
                
                progressContainer.appendChild(progressBar);
                probDiv.appendChild(label);
                probDiv.appendChild(progressContainer);
                probabilities.appendChild(probDiv);
            });

        } catch (error) {
            alert('Erreur lors de la classification: ' + error.message);
        }
    });
});
