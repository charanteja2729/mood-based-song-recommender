document.addEventListener('DOMContentLoaded', () => {

    // --- Get All DOM Elements ---
    const moodForm = document.getElementById('moodForm');
    const submitButton = document.getElementById('submitButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const buttonText = document.getElementById('buttonText');
    
    const resultsArea = document.getElementById('resultsArea');
    const detectedMood = document.getElementById('detectedMood');
    const songGrid = document.getElementById('songGrid');
    const errorMessage = document.getElementById('errorMessage');

    // Single audio player for the whole page
    const audioPlayer = document.getElementById('audioPlayer');
    
    // State variable to track the currently playing button
    let currentPlayingButton = null;

    // --- Form Submission Handler ---
    moodForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // Prevent default form submission

        // 1. Set Loading State
        setLoading(true);
        errorMessage.classList.add('hidden');
        resultsArea.classList.add('hidden');
        songGrid.innerHTML = ''; // Clear previous results

        // 2. Show Skeleton Loaders
        for (let i = 0; i < 8; i++) {
            songGrid.appendChild(createSkeletonCard());
        }
        resultsArea.classList.remove('hidden');
        detectedMood.textContent = 'Analyzing...';
        
        try {
            // 3. Get Form Data
            const formData = new FormData(moodForm);
            const data = {
                message: formData.get('message'),
                language: formData.get('language'),
                preference: formData.get('preference')
            };

            // 4. Call Backend API
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const result = await response.json();
            
            // 5. Display Results
            displayResults(result);

        } catch (error) {
            // 6. Display Error
            console.error('Error:', error);
            showError(error.message);
        } finally {
            // 7. Unset Loading State
            setLoading(false);
        }
    });

    // --- UI Helper Functions ---

    function setLoading(isLoading) {
        if (isLoading) {
            submitButton.disabled = true;
            loadingSpinner.classList.remove('hidden');
            buttonText.textContent = 'Loading...';
        } else {
            submitButton.disabled = false;
            loadingSpinner.classList.add('hidden');
            buttonText.textContent = 'Get Recommendations';
        }
    }

    function showError(message) {
        resultsArea.classList.add('hidden');
        errorMessage.textContent = `Error: ${message}`;
        errorMessage.classList.remove('hidden');
    }

    function displayResults(data) {
        // Clear skeleton loaders
        songGrid.innerHTML = '';
        errorMessage.classList.add('hidden');

        if (data.error) {
            showError(data.error);
            return;
        }

        // Update mood
        detectedMood.textContent = data.mood;
        resultsArea.classList.remove('hidden');

        // Create and append song cards
        if (data.songs && data.songs.length > 0) {
            data.songs.forEach(song => {
                const card = createSongCard(song);
                songGrid.appendChild(card);
            });
        } else {
            songGrid.innerHTML = '<p class="text-gray-500 col-span-full text-center">No songs found for this mood.</p>';
        }
    }

    function createSongCard(song) {
        const card = document.createElement('div');
        card.className = 'song-card bg-white rounded-lg shadow-md overflow-hidden';

        const image = document.createElement('img');
        image.src = song.image_url || 'https://placehold.co/300x300/e2e8f0/cbd5e0?text=Music';
        image.alt = song.song_name;
        image.className = 'w-full h-auto object-cover aspect-square';

        const info = document.createElement('div');
        info.className = 'p-3';

        const title = document.createElement('h3');
        title.textContent = song.song_name;
        title.className = 'font-semibold text-gray-800 truncate';

        const artist = document.createElement('p');
        artist.textContent = song.artist;
        artist.className = 'text-sm text-gray-500 truncate';
        
        // Play button overlay
        const overlay = document.createElement('div');
        overlay.className = 'play-button-overlay';
        
        // Play button
        const playButton = document.createElement('button');
        playButton.className = 'play-button';
        playButton.innerHTML = getPlayIcon(); // SVG for play

        if (song.preview_url) {
            playButton.addEventListener('click', (e) => {
                e.stopPropagation(); // Stop click from bubbling to card
                togglePlay(song.preview_url, playButton, card);
            });
        } else {
            playButton.disabled = true;
            playButton.style.opacity = '0.3';
        }

        overlay.appendChild(playButton);
        card.appendChild(image);
        card.appendChild(info);
        info.appendChild(title);
        info.appendChild(artist);
        card.appendChild(overlay);

        // Make the whole card link to Spotify
        card.addEventListener('click', () => {
            window.open(song.spotify_url, '_blank');
        });

        return card;
    }

    function createSkeletonCard() {
        const card = document.createElement('div');
        card.className = 'skeleton-card bg-white rounded-lg shadow-md overflow-hidden';
        
        const image = document.createElement('div');
        image.className = 'w-full bg-gray-300 aspect-square';
        
        const info = document.createElement('div');
        info.className = 'p-3 space-y-2';
        
        const title = document.createElement('div');
        title.className = 'h-4 bg-gray-300 rounded w-3/4';
        
        const artist = document.createElement('div');
        artist.className = 'h-4 bg-gray-300 rounded w-1/2';
        
        info.appendChild(title);
        info.appendChild(artist);
        card.appendChild(image);
        card.appendChild(info);
        
        return card;
    }

    // --- Audio Playback Logic ---

    function togglePlay(previewUrl, button, card) {
        if (audioPlayer.src === previewUrl && !audioPlayer.paused) {
            // Song is currently playing and it's the same song -> Pause it
            audioPlayer.pause();
        } else {
            // Song is paused or it's a new song -> Play it
            if (currentPlayingButton) {
                // Stop the old song/reset its button
                currentPlayingButton.innerHTML = getPlayIcon();
                currentPlayingButton.closest('.song-card').classList.remove('playing');
            }
            
            audioPlayer.src = previewUrl;
            audioPlayer.play();
            button.innerHTML = getPauseIcon(); // Set to pause icon
            card.classList.add('playing');
            currentPlayingButton = button;
        }
    }

    // Update button icons based on audio player events
    audioPlayer.addEventListener('pause', () => {
        if (currentPlayingButton) {
            currentPlayingButton.innerHTML = getPlayIcon();
            currentPlayingButton.closest('.song-card').classList.remove('playing');
            currentPlayingButton = null;
        }
    });

    audioPlayer.addEventListener('ended', () => {
        if (currentPlayingButton) {
            currentPlayingButton.innerHTML = getPlayIcon();
            currentPlayingButton.closest('.song-card').classList.remove('playing');
            currentPlayingButton = null;
        }
    });

    // --- SVG Icons ---
    function getPlayIcon() {
        return '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"></path></svg>';
    }

    function getPauseIcon() {
        return '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"></path></svg>';
    }
});
