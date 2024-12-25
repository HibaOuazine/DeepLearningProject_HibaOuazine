document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const resultsSection = document.getElementById('results');
    const mainPrediction = document.getElementById('main-prediction');
    const confidence = document.getElementById('confidence');
    const probabilities = document.getElementById('probabilities');
    const dropZone = document.getElementById('drop-zone');

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('highlight');
    }

    function unhighlight(e) {
        dropZone.classList.remove('highlight');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        imageInput.files = files;
        handleFiles(files);
    }

    // Preview image before upload
    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                    dropZone.querySelector('.upload-prompt').style.display = 'none';
                }
                reader.readAsDataURL(file);
            } else {
                alert('Veuillez sélectionner une image valide');
            }
        }
    }

    imageInput.addEventListener('change', function(e) {
        handleFiles(this.files);
    });

    // Handle form submission with loading state
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData();
        const imageFile = imageInput.files[0];
        
        if (!imageFile) {
            alert('Veuillez sélectionner une image');
            return;
        }

        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Classification en cours...';
        submitButton.disabled = true;

        formData.append('image', imageFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.error) {
                alert('Erreur: ' + result.error);
                return;
            }

            // Display results with animation
            resultsSection.style.display = 'block';
            mainPrediction.textContent = result.class_name.charAt(0).toUpperCase() + result.class_name.slice(1);
            confidence.textContent = (result.confidence * 100).toFixed(2);

            // Display all probabilities with animated bars
            probabilities.innerHTML = '';
            result.all_probabilities.forEach(prob => {
                const probDiv = document.createElement('div');
                probDiv.className = 'probability-item';
                
                const label = document.createElement('div');
                label.className = 'probability-label';
                label.innerHTML = `
                    <span>${prob.class.charAt(0).toUpperCase() + prob.class.slice(1)}</span>
                    <span>${(prob.probability * 100).toFixed(2)}%</span>
                `;
                
                const progressContainer = document.createElement('div');
                progressContainer.className = 'probability-bar';
                
                const progressBar = document.createElement('div');
                progressBar.className = 'probability-fill';
                
                // Animate the progress bar
                setTimeout(() => {
                    progressBar.style.width = `${prob.probability * 100}%`;
                }, 50);
                
                progressContainer.appendChild(progressBar);
                probDiv.appendChild(label);
                probDiv.appendChild(progressContainer);
                probabilities.appendChild(probDiv);
            });

            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            alert('Erreur lors de la classification: ' + error.message);
        } finally {
            // Restore button state
            submitButton.innerHTML = originalButtonText;
            submitButton.disabled = false;
        }
    });
});
