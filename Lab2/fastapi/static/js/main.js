document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const previewImage = document.getElementById('preview-image');
    const resultsSection = document.getElementById('results');
    const mainPrediction = document.getElementById('main-prediction');
    const confidence = document.getElementById('confidence');
    const probabilities = document.getElementById('probabilities');
    const dropZone = document.getElementById('drop-zone');

    // Initialize results card
    resultsSection.style.display = 'block';
    resultsSection.style.visibility = 'hidden';

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
        
        if (files && files.length > 0) {
            imageInput.files = files;
            handleFiles(files);
        }
    }

    // Preview image before upload
    function handleFiles(files) {
        if (files && files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                    const uploadPrompt = dropZone.querySelector('.upload-prompt');
                    if (uploadPrompt) {
                        uploadPrompt.style.display = 'none';
                    }
                };
                reader.onerror = function(e) {
                    console.error('FileReader error:', e);
                    alert('Erreur lors de la lecture du fichier');
                };
                reader.readAsDataURL(file);
            } else {
                alert('Veuillez sélectionner une image valide (JPG, PNG, etc.)');
            }
        }
    }

    imageInput.addEventListener('change', function(e) {
        handleFiles(this.files);
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

        // Show loading state
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Classification en cours...';
        submitButton.disabled = true;

        try {
            formData.append('file', imageFile);

            console.log('Sending request to /predict...');
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('La réponse du serveur n\'est pas au format JSON');
            }

            const result = await response.json();
            console.log('Received result:', result);

            if (result.error) {
                throw new Error(result.error);
            }

            // Show results section
            resultsSection.style.visibility = 'visible';
            resultsSection.classList.add('show');

            // Clear previous results
            mainPrediction.textContent = '';
            confidence.textContent = '';
            probabilities.innerHTML = '';

            // Update main prediction
            mainPrediction.textContent = result.class_name.charAt(0).toUpperCase() + result.class_name.slice(1);
            confidence.textContent = (result.confidence * 100).toFixed(2);

            // Update probabilities
            if (Array.isArray(result.all_probabilities)) {
                result.all_probabilities.forEach(prob => {
                    if (!prob || typeof prob.class !== 'string' || typeof prob.probability !== 'number') {
                        console.error('Invalid probability object:', prob);
                        return;
                    }

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
                    
                    setTimeout(() => {
                        progressBar.style.width = `${prob.probability * 100}%`;
                    }, 50);
                    
                    progressContainer.appendChild(progressBar);
                    probDiv.appendChild(label);
                    probDiv.appendChild(progressContainer);
                    probabilities.appendChild(probDiv);
                });
            } else {
                console.error('all_probabilities is not an array:', result.all_probabilities);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('Erreur: ' + error.message);
            resultsSection.style.visibility = 'hidden';
            resultsSection.classList.remove('show');
        } finally {
            submitButton.innerHTML = originalButtonText;
            submitButton.disabled = false;
        }
    });
});
