document.addEventListener('DOMContentLoaded', () => {
    // State management
    const state = {
        currentStep: 1,
        totalSteps: 5,
        originalImage: null,
        detectedParts: [],
        selectedPart: null,
        isProcessing: false
    };

    // DOM Elements
    const imageUpload = document.getElementById('imageUpload');
    const replacementUpload = document.getElementById('replacementUpload');
    const detectButton = document.getElementById('detectButton');
    const segmentButton = document.getElementById('segmentButton');
    const replaceButton = document.getElementById('replaceButton');
    const prevButton = document.getElementById('prevButton');
    const nextButton = document.getElementById('nextButton');
    const progressBar = document.getElementById('progress-bar');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadingText = document.getElementById('loadingText');

    // Event Listeners
    imageUpload.addEventListener('change', handleImageUpload);
    detectButton.addEventListener('click', detectParts);
    segmentButton.addEventListener('click', applySegmentation);
    replaceButton.addEventListener('click', replacePart);
    prevButton.addEventListener('click', () => navigateStep(-1));
    nextButton.addEventListener('click', () => navigateStep(1));

    // Loading State Management
    function showLoading(message = 'Processing...') {
        state.isProcessing = true;
        loadingText.textContent = message;
        loadingOverlay.style.display = 'flex';
        disableButtons(true);
    }

    function hideLoading() {
        state.isProcessing = false;
        loadingOverlay.style.display = 'none';
        disableButtons(false);
    }

    function disableButtons(disabled) {
        detectButton.disabled = disabled;
        segmentButton.disabled = disabled;
        replaceButton.disabled = disabled;
        prevButton.disabled = disabled || state.currentStep === 1;
        nextButton.disabled = disabled || state.currentStep === state.totalSteps;
    }

    // Navigation Functions
    function navigateStep(direction) {
        if (state.isProcessing) return;
        
        const newStep = state.currentStep + direction;
        if (newStep >= 1 && newStep <= state.totalSteps) {
            document.querySelector(`#step${state.currentStep}`).classList.remove('active');
            document.querySelector(`#step${newStep}`).classList.add('active');
            state.currentStep = newStep;
            updateProgress();
            updateNavigationButtons();
        }
    }

    function updateProgress() {
        const progress = (state.currentStep / state.totalSteps) * 100;
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `Step ${state.currentStep}: ${getStepName(state.currentStep)}`;
    }

    function getStepName(step) {
        const steps = {
            1: 'Upload Image',
            2: 'Detect Parts',
            3: 'Segmentation',
            4: 'Extract Parts',
            5: 'Replace Part'
        };
        return steps[step];
    }

    function updateNavigationButtons() {
        disableButtons(state.isProcessing);
    }

    // Image Processing Functions
    async function handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        showLoading('Loading image...');
        try {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = document.createElement('img');
                img.onload = () => {
                    state.originalImage = e.target.result;
                    const container = document.getElementById('originalImageContainer');
                    container.innerHTML = '';
                    container.appendChild(img.cloneNode());
                    nextButton.disabled = false;
                    hideLoading();
                };
                img.onerror = () => {
                    throw new Error('Failed to load image');
                };
                img.src = e.target.result;
            };
            reader.onerror = () => {
                throw new Error('Failed to read file');
            };
            reader.readAsDataURL(file);
        } catch (error) {
            console.error('Error loading image:', error);
            alert('Error loading image. Please try again.');
            hideLoading();
        }
    }

    async function detectParts() {
        if (!state.originalImage || state.isProcessing) return;

        showLoading('Detecting car parts...');
        try {
            const formData = new FormData();
            formData.append('image', dataURLtoBlob(state.originalImage));

            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (!data.success) {
                throw new Error(data.error || 'Detection failed');
            }

            const container = document.getElementById('detectedImageContainer');
            const img = new Image();
            img.onload = () => {
                container.innerHTML = '';
                container.appendChild(img);
                state.detectedParts = data.detected_parts;
                
                if (state.detectedParts.length === 0) {
                    alert('No car parts were detected in the image. Please try with a different image.');
                    hideLoading();
                    return;
                }
                
                nextButton.disabled = false;
                hideLoading();
            };
            img.onerror = () => {
                throw new Error('Failed to load detected image');
            };
            img.src = `data:image/jpeg;base64,${data.detected_image}`;
        } catch (error) {
            console.error('Error detecting parts:', error);
            alert(`Error detecting parts: ${error.message}`);
            hideLoading();
        }
    }

    async function applySegmentation() {
        if (!state.originalImage || !state.detectedParts.length || state.isProcessing) return;

        showLoading('Applying segmentation...');
        try {
            const formData = new FormData();
            formData.append('image', dataURLtoBlob(state.originalImage));
            formData.append('parts', JSON.stringify(state.detectedParts));

            const response = await fetch('/segment', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (!data.success) {
                throw new Error(data.error || 'Segmentation failed');
            }

            const container = document.getElementById('segmentedImageContainer');
            const img = new Image();
            img.onload = () => {
                container.innerHTML = '';
                container.appendChild(img);
                
                // Create color legend
                const legend = document.getElementById('colorLegend');
                legend.innerHTML = '';
                data.colors.forEach(({label, color}) => {
                    const item = document.createElement('div');
                    item.className = 'color-item';
                    item.innerHTML = `
                        <div class="color-box" style="background-color: rgb(${color.map(c => Math.round(c * 255)).join(',')})"></div>
                        <span>${label}</span>
                    `;
                    legend.appendChild(item);
                });

                // Create part buttons
                const buttonsContainer = document.getElementById('partButtons');
                buttonsContainer.innerHTML = '';
                state.detectedParts.forEach((part, index) => {
                    const button = document.createElement('button');
                    button.className = 'btn btn-outline-primary part-button';
                    button.textContent = part.label;
                    button.onclick = () => extractPart(index);
                    buttonsContainer.appendChild(button);
                });

                nextButton.disabled = false;
                hideLoading();
            };
            img.onerror = () => {
                throw new Error('Failed to load segmented image');
            };
            img.src = `data:image/jpeg;base64,${data.segmented_image}`;
        } catch (error) {
            console.error('Error applying segmentation:', error);
            alert(`Error applying segmentation: ${error.message}`);
            hideLoading();
        }
    }

    async function extractPart(partIndex) {
        if (!state.originalImage || !state.detectedParts[partIndex] || state.isProcessing) return;

        showLoading('Extracting part...');
        try {
            const formData = new FormData();
            formData.append('image', dataURLtoBlob(state.originalImage));
            formData.append('part_index', partIndex);
            
            // Add bounding box coordinates
            const bbox = state.detectedParts[partIndex].bbox;
            formData.append('bbox[0]', bbox[0]);
            formData.append('bbox[1]', bbox[1]);
            formData.append('bbox[2]', bbox[2]);
            formData.append('bbox[3]', bbox[3]);

            const response = await fetch('/extract', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (!data.success) {
                throw new Error(data.error || 'Extraction failed');
            }

            state.selectedPart = partIndex;
            
            const extractedContainer = document.getElementById('extractedPartContainer');
            const maskedContainer = document.getElementById('maskedOriginalContainer');
            
            const extractedImg = new Image();
            const maskedImg = new Image();
            
            let loadedImages = 0;
            const onImageLoad = () => {
                loadedImages++;
                if (loadedImages === 2) {
                    extractedContainer.innerHTML = '';
                    maskedContainer.innerHTML = '';
                    extractedContainer.appendChild(extractedImg);
                    maskedContainer.appendChild(maskedImg);
                    nextButton.disabled = false;
                    hideLoading();
                }
            };

            extractedImg.onload = onImageLoad;
            maskedImg.onload = onImageLoad;

            extractedImg.src = `data:image/jpeg;base64,${data.extracted_part}`;
            maskedImg.src = `data:image/jpeg;base64,${data.masked_original}`;
        } catch (error) {
            console.error('Error extracting part:', error);
            alert(`Error extracting part: ${error.message}`);
            hideLoading();
        }
    }

    async function replacePart() {
        const file = replacementUpload.files[0];
        if (!file || state.selectedPart === null || state.isProcessing) return;

        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        showLoading('Replacing part...');
        try {
            const formData = new FormData();
            formData.append('original_image', dataURLtoBlob(state.originalImage));
            formData.append('replacement_image', file);
            formData.append('part_index', state.selectedPart);

            // Add bounding box coordinates
            const bbox = state.detectedParts[state.selectedPart].bbox;
            formData.append('bbox[0]', bbox[0]);
            formData.append('bbox[1]', bbox[1]);
            formData.append('bbox[2]', bbox[2]);
            formData.append('bbox[3]', bbox[3]);

            const response = await fetch('/replace', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }

            if (!data.success) {
                throw new Error(data.error || 'Replacement failed');
            }

            const container = document.getElementById('replacedImageContainer');
            const img = new Image();
            img.onload = () => {
                container.innerHTML = '';
                container.appendChild(img);
                hideLoading();
            };
            img.onerror = () => {
                throw new Error('Failed to load result image');
            };
            img.src = `data:image/jpeg;base64,${data.result_image}`;
        } catch (error) {
            console.error('Error replacing part:', error);
            alert(`Error replacing part: ${error.message}`);
            hideLoading();
        }
    }

    // Utility function to convert Data URL to Blob
    function dataURLtoBlob(dataURL) {
        const arr = dataURL.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new Blob([u8arr], { type: mime });
    }
}); 