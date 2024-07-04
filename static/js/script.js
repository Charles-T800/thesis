// script.js

// JavaScript function to refresh the page
function refreshPage() {
    location.reload();
}

// Function to display the error message
function showError() {
    // Show the error message div
    document.getElementById('error-message').style.display = 'block';
}

// Function to hide the error message
function hideError() {
    // Hide the error message div
    document.getElementById('error-message').style.display = 'none';
}

// Function to handle face detection success or failure
function detectFaceFailure() {
    // Check if there was an error loading the video feed
    const videoFeedError = document.getElementById('video-feed').naturalWidth === 0;

    if (videoFeedError) {
        // Show the error message if there was an error loading the video feed
        showError();
    } else {
        // Hide the error message if face detection was successful
        hideError();
    }
}

// Call the detectFaceFailure() function when the page loads (or at the appropriate time)
window.onload = detectFaceFailure;

// Check if the video feed has loaded successfully
document.getElementById('video-feed').onload = () => {
    console.log('Video feed loaded');
    // Show the refresh button
    document.getElementById('refresh-button-container').style.display = 'block';
};

// Handle error if face detection fails
document.getElementById('video-feed').onerror = () => {
    console.error('Error loading video feed');
    // Show the error message
    document.getElementById('error-message').style.display = 'block';
};

function toggleRefreshButton() {
    const username = document.getElementById('username').value;
    const refreshButtonContainer = document.getElementById('refresh-button-container');
    const errorMessage = document.getElementById('error-message');

    // Check if username is not empty
    if (username.trim() !== '') {
        refreshButtonContainer.style.display = 'block';
        errorMessage.style.display = 'none'; // Hide error message if shown
    } else {
        refreshButtonContainer.style.display = 'none';
    }
}

function refreshPage() {
    location.reload();
}

