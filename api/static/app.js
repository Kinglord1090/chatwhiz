/**
 * ChatWhiz Modern Frontend Application
 * Features: Live progress updates, persistent state, dark mode, responsive design
 */

class ChatWhizModern {
    constructor() {
        this.apiUrl = '';
        this.currentTask = null;
        this.pollInterval = null;
        this.lastSearchQuery = null;
        this.indexingState = null;
        this.allSearchResults = [];  // Store all search results
        this.currentPage = 1;
        this.resultsPerPage = 5;
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing ChatWhiz Modern...');
        
        // Check for any ongoing indexing tasks
        await this.checkOngoingTasks();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize system status
        await this.loadSystemStatus();
        
        // Setup auto-refresh for status
        setInterval(() => this.loadSystemStatus(), 10000);
        
        // Load saved state
        this.loadSavedState();
        
        console.log('✅ ChatWhiz Modern initialized!');
    }

    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.closest('.tab-btn').dataset.tab);
            });
        });

        // Search functionality
        const searchBtn = document.getElementById('search-btn');
        const searchQuery = document.getElementById('search-query');
        
        searchBtn?.addEventListener('click', () => this.performSearch());
        searchQuery?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.performSearch();
        });

        // AI Assistant
        const aiAskBtn = document.getElementById('ai-ask-btn');
        const aiPrompt = document.getElementById('ai-prompt');
        
        aiAskBtn?.addEventListener('click', () => this.askAI());
        aiPrompt?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !aiAskBtn.disabled) this.askAI();
        });

        // File upload
        const fileInput = document.getElementById('file-input');
        const dropZone = document.getElementById('drop-zone');
        const uploadBtn = document.getElementById('upload-btn');
        
        dropZone?.addEventListener('click', () => fileInput.click());
        fileInput?.addEventListener('change', (e) => this.handleFileSelection(e.target.files));
        uploadBtn?.addEventListener('click', () => this.uploadFiles());

        // Drag and drop
        if (dropZone) {
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('border-accent-primary', 'bg-dark-hover');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('border-accent-primary', 'bg-dark-hover');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-accent-primary', 'bg-dark-hover');
                this.handleFileSelection(e.dataTransfer.files);
            });
        }

        // Dark mode toggle removed - not implemented
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('tab-active', 'bg-dark-hover');
        });
        
        const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
        if (activeTab) {
            activeTab.classList.add('tab-active', 'bg-dark-hover');
        }

        // Show/hide tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        
        const tabContent = document.getElementById(`${tabName}-tab`);
        if (tabContent) {
            tabContent.classList.remove('hidden');
            tabContent.classList.add('animate-fade-in');
        }

        // Load tab-specific data
        this.loadTabData(tabName);
    }

    async loadTabData(tabName) {
        switch (tabName) {
            case 'upload':
                await this.updateUploadStats();
                break;
            case 'analytics':
                await this.loadAnalyticsData();
                break;
        }
    }

    async checkOngoingTasks() {
        try {
            // First check for any ongoing task from localStorage
            const savedTask = localStorage.getItem('chatwhiz_current_task');
            if (savedTask) {
                const task = JSON.parse(savedTask);
                console.log(`Checking saved task ${task.task_id}...`);
                const response = await fetch(`/api/task/${task.task_id}`);
                if (response.ok) {
                    const taskStatus = await response.json();
                    console.log(`Task ${task.task_id} status: ${taskStatus.status}`);
                    if (taskStatus.status === 'started' || taskStatus.status === 'processing' || taskStatus.status === 'queued') {
                        console.log(`Resuming task ${task.task_id} with progress ${taskStatus.progress}`);
                        this.currentTask = task.task_id;
                        this.switchTab('upload');
                        this.resumeProgressTracking();
                        return; // Exit if we found a task to resume
                    } else {
                        // Task is completed or errored, clear it
                        console.log(`Task ${task.task_id} is ${taskStatus.status}, clearing from localStorage`);
                        localStorage.removeItem('chatwhiz_current_task');
                    }
                } else {
                    // Task not found, clear localStorage
                    console.log(`Task ${task.task_id} not found on server, clearing localStorage`);
                    localStorage.removeItem('chatwhiz_current_task');
                }
            }
            
            // Check if server has any active tasks (including resumed ones)
            console.log('Checking for active tasks on server...');
            const statusResponse = await fetch('/api/status');
            if (statusResponse.ok) {
                const status = await statusResponse.json();
                if (status.active_tasks && status.active_tasks.length > 0) {
                    // Server has active tasks
                    const activeTask = status.active_tasks[0];
                    console.log(`Found active task on server: ${activeTask.task_id}`);
                    this.currentTask = activeTask.task_id;
                    // Save it to localStorage
                    localStorage.setItem('chatwhiz_current_task', JSON.stringify({
                        task_id: activeTask.task_id,
                        timestamp: Date.now()
                    }));
                    // Show upload tab and start tracking
                    this.switchTab('upload');
                    this.resumeProgressTracking();
                } else {
                    console.log('No active tasks found on server');
                }
            }
        } catch (error) {
            console.error('Error checking ongoing tasks:', error);
            localStorage.removeItem('chatwhiz_current_task');
        }
    }

    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            this.updateSystemIndicators(status);
            this.updateStats(status.stats);
            
            if (status.config?.available_llms) {
                this.updateLLMOptions(status.config.available_llms);
            }
        } catch (error) {
            console.error('Failed to load system status:', error);
            this.updateSystemIndicators({ initialized: false });
        }
    }

    updateSystemIndicators(status) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');
        
        if (status.initialized) {
            indicator?.classList.replace('bg-yellow-500', 'bg-green-500');
            if (text) text.textContent = 'Ready';
        } else {
            indicator?.classList.replace('bg-green-500', 'bg-yellow-500');
            if (text) text.textContent = 'Initializing';
        }
    }

    updateStats(stats) {
        if (!stats) return;
        
        // Update component status dimension and index type
        const componentDim = document.getElementById('component-dimension');
        if (componentDim) componentDim.textContent = stats.embedding_dimension || 0;
        
        const componentIndexType = document.getElementById('component-index-type');
        if (componentIndexType) componentIndexType.textContent = stats.index_type || 'flat';
    }

    updateLLMOptions(availableLLMs) {
        const llmProvider = document.getElementById('llm-provider');
        if (!llmProvider) return;
        
        // Store current selection if any
        const currentSelection = llmProvider.value;
        
        llmProvider.innerHTML = '';
        availableLLMs.forEach(llm => {
            const option = document.createElement('option');
            option.value = llm;
            // Handle both instructor-large and instructor-qa naming
            if (llm === 'instructor-large' || llm === 'instructor-qa') {
                option.textContent = 'Instructor Q&A';
            } else {
                option.textContent = llm;
            }
            llmProvider.appendChild(option);
        });
        
        // Restore selection if it still exists
        if (currentSelection && availableLLMs.includes(currentSelection)) {
            llmProvider.value = currentSelection;
        }
    }

    async performSearch() {
        const query = document.getElementById('search-query')?.value.trim();
        if (!query) {
            this.showToast('Please enter a search query', 'warning');
            return;
        }

        const searchBtn = document.getElementById('search-btn');
        const originalText = searchBtn.innerHTML;
        
        searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Searching...';
        searchBtn.disabled = true;

        try {
            const searchData = {
                query: query,
                mode: document.getElementById('search-mode')?.value || 'semantic',
                top_k: 1000,  // Get up to 1000 results (essentially all relevant results)
                threshold: 0.3  // Lower threshold to get more results
            };

            const response = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(searchData)
            });

            if (!response.ok) throw new Error('Search failed');

            const result = await response.json();
            
            // Store all results and reset to page 1
            this.allSearchResults = result.results || [];
            this.currentPage = 1;
            this.lastSearchQuery = query;
            
            // Display first page of results
            this.displaySearchResultsPage();
            
            // Save search to localStorage for persistence
            localStorage.setItem('chatwhiz_last_search', query);
            localStorage.setItem('chatwhiz_last_results', JSON.stringify(result));
            
            // Enable AI assistant with all results
            const aiAskBtn = document.getElementById('ai-ask-btn');
            if (aiAskBtn) aiAskBtn.disabled = false;
            
        } catch (error) {
            console.error('Search failed:', error);
            this.showToast('Search failed. Please try again.', 'error');
        } finally {
            searchBtn.innerHTML = originalText;
            searchBtn.disabled = false;
        }
    }
    displaySearchResultsPage() {
        const resultsSection = document.getElementById('search-results');
        const resultsContainer = document.getElementById('results-container');
        const resultsCount = document.getElementById('results-count');
        const aiAssistant = document.getElementById('ai-assistant');
        
        if (!this.allSearchResults || this.allSearchResults.length === 0) {
            resultsContainer.innerHTML = `
                <div class="text-center py-12 text-dark-muted">
                    <i class="fas fa-search-minus text-4xl mb-4 opacity-50"></i>
                    <p>No results found. Try adjusting your query or search mode.</p>
                </div>
            `;
            resultsCount.textContent = '0 results';
            aiAssistant?.classList.add('hidden');
        } else {
            // Calculate pagination
            const totalResults = this.allSearchResults.length;
            const totalPages = Math.ceil(totalResults / this.resultsPerPage);
            const startIndex = (this.currentPage - 1) * this.resultsPerPage;
            const endIndex = Math.min(startIndex + this.resultsPerPage, totalResults);
            const pageResults = this.allSearchResults.slice(startIndex, endIndex);
            
            // Update count
            resultsCount.textContent = `${totalResults} results found (showing ${startIndex + 1}-${endIndex})`;
            
            // Display current page results
            let html = pageResults.map((result, index) => `
                <div class="glass-card rounded-lg p-4 glass-card-hover animate-slide-up" style="animation-delay: ${index * 0.05}s">
                    <div class="flex items-start justify-between mb-2">
                        <div class="flex items-center space-x-2">
                            <span class="text-sm font-medium text-accent-primary">#${startIndex + index + 1}</span>
                            <span class="px-2 py-0.5 bg-accent-${this.getSearchTypeColor(result.search_type)}/20 text-accent-${this.getSearchTypeColor(result.search_type)} text-xs rounded">
                                ${result.search_type}
                            </span>
                        </div>
                        <span class="text-xs font-mono text-dark-muted">${result.score.toFixed(4)}</span>
                    </div>
                    <div class="text-dark-text leading-relaxed">${this.escapeHtml(result.text)}</div>
                    ${result.metadata ? this.renderMetadata(result.metadata) : ''}
                </div>
            `).join('');
            
            // Add pagination controls
            if (totalPages > 1) {
                html += `
                    <div class="flex items-center justify-center space-x-2 mt-6">
                        <button 
                            onclick="window.chatwhiz.changePage(-1)" 
                            class="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:bg-dark-hover transition-all disabled:opacity-50"
                            ${this.currentPage === 1 ? 'disabled' : ''}
                        >
                            <i class="fas fa-chevron-left"></i>
                        </button>
                        <span class="px-4 py-2 text-sm">
                            Page ${this.currentPage} of ${totalPages}
                        </span>
                        <button 
                            onclick="window.chatwhiz.changePage(1)" 
                            class="px-4 py-2 bg-dark-card border border-dark-border rounded-lg hover:bg-dark-hover transition-all disabled:opacity-50"
                            ${this.currentPage === totalPages ? 'disabled' : ''}
                        >
                            <i class="fas fa-chevron-right"></i>
                        </button>
                    </div>
                `;
            }
            
            resultsContainer.innerHTML = html;
            
            // Enable AI assistant with ALL results (not just current page)
            aiAssistant?.classList.remove('hidden');
            const aiAskBtn = document.getElementById('ai-ask-btn');
            if (aiAskBtn) aiAskBtn.disabled = false;
        }
        
        resultsSection?.classList.remove('hidden');
        resultsSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    changePage(direction) {
        const totalPages = Math.ceil(this.allSearchResults.length / this.resultsPerPage);
        const newPage = this.currentPage + direction;
        
        if (newPage >= 1 && newPage <= totalPages) {
            this.currentPage = newPage;
            this.displaySearchResultsPage();
        }
    }

    // Keep old method for compatibility
    displaySearchResults(results) {
        // Store all results and display first page
        this.allSearchResults = results.results || [];
        this.currentPage = 1;
        this.displaySearchResultsPage();
    }

    getSearchTypeColor(type) {
        switch (type) {
            case 'semantic': return 'primary';
            case 'bm25': return 'secondary';
            case 'hybrid': return 'success';
            default: return 'info';
        }
    }

    renderMetadata(metadata) {
        const fields = ['sender', 'timestamp', 'message_id'];
        const metadataHtml = fields
            .filter(field => metadata[field])
            .map(field => `<span class="text-xs text-dark-muted">${field}: ${this.escapeHtml(String(metadata[field]))}</span>`)
            .join(' • ');
        
        return metadataHtml ? `<div class="mt-2 pt-2 border-t border-dark-border">${metadataHtml}</div>` : '';
    }

    async askAI() {
        const prompt = document.getElementById('ai-prompt')?.value.trim();
        const query = prompt || this.lastSearchQuery;
        
        if (!query) {
            this.showToast('Please enter a question', 'warning');
            return;
        }

        const aiAskBtn = document.getElementById('ai-ask-btn');
        const originalText = aiAskBtn.innerHTML;
        
        aiAskBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Thinking...';
        aiAskBtn.disabled = true;

        try {
            // Use ALL search results for context, not just the displayed page
            const ragData = {
                query: query,
                mode: document.getElementById('search-mode')?.value || 'semantic',
                top_k: Math.max(this.allSearchResults.length, 100),  // Use all results or at least 100
                threshold: 0.3,
                llm_provider: document.getElementById('llm-provider')?.value || 'instructor-large'
            };

            const response = await fetch('/api/rag', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(ragData)
            });

            if (!response.ok) throw new Error('AI response failed');

            const result = await response.json();
            this.displayAIResponse(result);
            
        } catch (error) {
            console.error('AI response failed:', error);
            this.showToast('AI response generation failed', 'error');
        } finally {
            aiAskBtn.innerHTML = originalText;
            aiAskBtn.disabled = false;
        }
    }

    displayAIResponse(result) {
        const aiResponse = document.getElementById('ai-response');
        const aiAnswer = document.getElementById('ai-answer');
        const aiMetadata = document.getElementById('ai-metadata');
        
        if (aiAnswer) {
            aiAnswer.innerHTML = this.formatAIResponse(result.answer);
        }
        
        if (aiMetadata) {
            aiMetadata.innerHTML = `
                LLM: ${result.llm_used} • 
                Context: ${result.context_used || result.results?.length || 0} messages
                ${result.error ? ` • Error: ${result.error}` : ''}
            `;
        }
        
        aiResponse?.classList.remove('hidden');
    }

    formatAIResponse(text) {
        return text.split('\n\n').map(paragraph => 
            `<p class="mb-3">${this.escapeHtml(paragraph).replace(/\n/g, '<br>')}</p>`
        ).join('');
    }

    handleFileSelection(files) {
        const selectedFiles = document.getElementById('selected-files');
        const fileList = document.getElementById('file-list');
        const uploadBtn = document.getElementById('upload-btn');
        
        if (files.length > 0) {
            selectedFiles?.classList.remove('hidden');
            if (uploadBtn) uploadBtn.disabled = false;
            
            if (fileList) {
                fileList.innerHTML = Array.from(files).map(file => `
                    <div class="glass-card rounded-lg p-3 flex items-center justify-between">
                        <div class="flex items-center space-x-3">
                            <i class="fas fa-file-${this.getFileIcon(file.name)} text-accent-info"></i>
                            <div>
                                <div class="text-sm font-medium">${file.name}</div>
                                <div class="text-xs text-dark-muted">${this.formatFileSize(file.size)}</div>
                            </div>
                        </div>
                        <span class="text-xs text-accent-success">Ready</span>
                    </div>
                `).join('');
            }
        } else {
            selectedFiles?.classList.add('hidden');
            if (uploadBtn) uploadBtn.disabled = true;
        }
    }

    getFileIcon(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        switch (ext) {
            case 'json': return 'code';
            case 'csv': return 'table';
            case 'txt': return 'alt';
            default: return 'file';
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async uploadFiles() {
        const fileInput = document.getElementById('file-input');
        const files = fileInput?.files;
        
        if (!files || files.length === 0) {
            this.showToast('Please select files to upload', 'warning');
            return;
        }

        const progressSection = document.getElementById('indexing-progress');
        const progressBar = document.getElementById('progress-bar');
        const progressPercent = document.getElementById('progress-percent');
        const progressMessage = document.getElementById('progress-message');
        const rebuildIndex = document.getElementById('rebuild-index')?.checked || false;
        
        progressSection?.classList.remove('hidden');
        
        // Show rebuild status if enabled
        if (rebuildIndex) {
            this.showToast('Rebuilding index - clearing cache...', 'info');
        }
        
        try {
            const formData = new FormData();
            Array.from(files).forEach(file => {
                formData.append('files', file);
            });

            // Add rebuild flag as query parameter
            const url = rebuildIndex ? '/api/upload?rebuild=true' : '/api/upload';
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Upload failed');

            const result = await response.json();
            // Handle multiple task IDs for multiple files
            const taskId = result.task_ids ? result.task_ids[0] : result.task_id;
            this.currentTask = taskId;
            
            // Save task to localStorage
            localStorage.setItem('chatwhiz_current_task', JSON.stringify({
                task_id: taskId,
                timestamp: Date.now()
            }));
            
            // Define the progress callback function
            const progressCallback = (progress) => {
                // Calculate percent correctly
                // Backend now sends overall progress as 0-1 (average of embedding and BM25)
                const percent = Math.min(100, (progress.progress || 0) * 100);
                
                if (progressBar) progressBar.style.width = `${percent}%`;
                if (progressPercent) progressPercent.textContent = `${percent.toFixed(1)}%`;
                
                // No need to extract count - it's already in the message
                
                // Show simplified message without redundant percentage
                if (progressMessage) {
                    // Remove percentage from message since it's shown separately
                    let msg = progress.message || 'Processing...';
                    msg = msg.replace(/\s*\(\d+%\)/, '');  // Remove (XX%) from message
                    progressMessage.textContent = msg;
                }
                
                if (progress.status === 'completed') {
                    // If this is the file-processing task finishing, it may hand off to an indexing task
                    const nextIndexTaskId = progress?.result?.index_task_id;
                    if (nextIndexTaskId) {
                        // Stop current tracking and switch to new task
                        clearInterval(this.pollInterval);
                        this.pollInterval = null;
                        this.currentTask = nextIndexTaskId;
                        localStorage.setItem('chatwhiz_current_task', JSON.stringify({
                            task_id: nextIndexTaskId,
                            timestamp: Date.now()
                        }));
                        if (progressMessage) progressMessage.textContent = 'Starting indexing...';
                        // Reset progress bar for new task
                        if (progressBar) progressBar.style.width = '0%';
                        if (progressPercent) progressPercent.textContent = '0%';
                        // Start tracking the new task with the same callback
                        setTimeout(() => this.trackProgress(progressCallback), 100);
                        return; // Exit this callback to avoid duplicate completion handling
                    }

                    // Otherwise, this is the final indexing completion
                    this.showToast('Files indexed successfully!', 'success');
                    this.updateUploadStats();
                    localStorage.removeItem('chatwhiz_current_task');
                    
                    setTimeout(() => {
                        progressSection?.classList.add('hidden');
                        if (fileInput) fileInput.value = '';
                        this.handleFileSelection([]);
                    }, 2000);
                } else if (progress.status === 'error') {
                    this.showToast(`Indexing failed: ${progress.message}`, 'error');
                    progressSection?.classList.add('hidden');
                    localStorage.removeItem('chatwhiz_current_task');
                }
            };
            
            // Start tracking with the callback
            this.trackProgress(progressCallback);
            
        } catch (error) {
            console.error('Upload failed:', error);
            this.showToast('Upload failed. Please try again.', 'error');
            progressSection?.classList.add('hidden');
        }
    }

    trackProgress(callback) {
        if (!this.currentTask) return;
        
        // Clear any existing interval
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }
        
        // Immediately fetch once before starting interval
        this.fetchProgress(callback);
        
        // Poll more frequently for granular updates
        this.pollInterval = setInterval(async () => {
            await this.fetchProgress(callback);
        }, 500); // Poll every 500ms for more granular updates
    }
    
    async fetchProgress(callback) {
        try {
            const response = await fetch(`/api/task/${this.currentTask}`);
            if (!response.ok) throw new Error('Task not found');
            
            const progress = await response.json();
            callback(progress);
            
            if (progress.status === 'completed' || progress.status === 'error') {
                // Don't clear if we're switching tasks
                if (!progress?.result?.index_task_id) {
                    clearInterval(this.pollInterval);
                    this.pollInterval = null;
                    this.currentTask = null;
                }
            }
        } catch (error) {
            console.error('Failed to track progress:', error);
            clearInterval(this.pollInterval);
            this.currentTask = null;
        }
    }
    
    async fetchAndDisplayProgress() {
        if (!this.currentTask) return;
        
        try {
            const response = await fetch(`/api/task/${this.currentTask}`);
            if (!response.ok) throw new Error('Task not found');
            
            const progress = await response.json();
            
            const progressBar = document.getElementById('progress-bar');
            const progressPercent = document.getElementById('progress-percent');
            const progressMessage = document.getElementById('progress-message');
            
            // Calculate percent correctly
            const percent = Math.min(100, (progress.progress || 0) * 100);
            
            if (progressBar) progressBar.style.width = `${percent}%`;
            if (progressPercent) progressPercent.textContent = `${percent.toFixed(1)}%`;
            
            if (progressMessage) {
                let msg = progress.message || 'Resuming...';
                msg = msg.replace(/\s*\(\d+%\)/, '');  // Remove percentage from message
                progressMessage.textContent = msg;
            }
        } catch (error) {
            console.error('Failed to fetch initial progress:', error);
        }
    }

    resumeProgressTracking() {
        const progressSection = document.getElementById('indexing-progress');
        progressSection?.classList.remove('hidden');
        
        // Immediately fetch and display current progress
        this.fetchAndDisplayProgress();
        
        this.trackProgress((progress) => {
            const progressBar = document.getElementById('progress-bar');
            const progressPercent = document.getElementById('progress-percent');
            const progressMessage = document.getElementById('progress-message');
            const progressSpinner = document.getElementById('progress-spinner');
            
            // Calculate percent correctly
            // Backend now sends overall progress as 0-1 (average of embedding and BM25)
            const percent = Math.min(100, (progress.progress || 0) * 100);
            
            if (progressBar) progressBar.style.width = `${percent}%`;
            if (progressPercent) progressPercent.textContent = `${percent.toFixed(1)}%`;
            
            // No need for progress details - count is in the message
            
            // Show simplified message
            if (progressMessage) {
                let msg = progress.message || 'Resuming...';
                msg = msg.replace(/\s*\(\d+%\)/, '');  // Remove percentage from message
                progressMessage.textContent = msg;
            }
            
            if (progress.status === 'completed') {
                this.showToast('Indexing completed!', 'success');
                this.updateUploadStats();
                localStorage.removeItem('chatwhiz_current_task');
                
                // Stop spinner on completion
                if (progressSpinner) progressSpinner.classList.remove('fa-spin');
                
                setTimeout(() => {
                    progressSection?.classList.add('hidden');
                }, 2000);
            } else if (progress.status === 'error') {
                this.showToast(`Indexing failed: ${progress.message}`, 'error');
                progressSection?.classList.add('hidden');
                localStorage.removeItem('chatwhiz_current_task');
                // Stop spinner on error
                if (progressSpinner) progressSpinner.classList.remove('fa-spin');
            }
        });
    }

    async updateUploadStats() {
        await this.loadSystemStatus();
    }

    async loadAnalyticsData() {
        await this.loadSystemStatus();
        
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            // Update model information
            if (status.config) {
                const modelName = document.getElementById('model-name');
                const modelDevice = document.getElementById('model-device');
                const modelLLM = document.getElementById('model-llm');
                
                if (modelName) modelName.textContent = status.config.embedding_model || 'Unknown';
                if (modelDevice) modelDevice.textContent = status.config.device || 'CPU';
                if (modelLLM) modelLLM.textContent = status.config.llm_provider || 'None';
            }
            
            // Update component status
            this.updateComponentStatus(status);
            
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }

    updateComponentStatus(status) {
        const components = {
            'status-embedder': status.initialized,
            'status-vector': status.stats?.semantic_vectors > 0,
            'status-bm25': status.stats?.bm25_documents > 0,
            'status-llm': status.config?.available_llms?.length > 0
        };
        
        Object.entries(components).forEach(([id, ready]) => {
            const element = document.getElementById(id);
            if (element) {
                if (ready) {
                    element.className = 'px-2 py-1 rounded text-xs bg-green-500/20 text-green-400';
                    element.textContent = 'Ready';
                } else {
                    element.className = 'px-2 py-1 rounded text-xs bg-yellow-500/20 text-yellow-400';
                    element.textContent = 'Not Ready';
                }
            }
        });
    }


    loadSavedState() {
        // Load last search
        const lastSearch = localStorage.getItem('chatwhiz_last_search');
        const lastResults = localStorage.getItem('chatwhiz_last_results');
        
        if (lastSearch) {
            const searchQuery = document.getElementById('search-query');
            if (searchQuery) {
                searchQuery.value = lastSearch;
                this.lastSearchQuery = lastSearch;
            }
            
            // Restore last search results if available
            if (lastResults) {
                try {
                    const results = JSON.parse(lastResults);
                    this.displaySearchResults(results);
                    
                    // Enable AI assistant since we have results
                    const aiAskBtn = document.getElementById('ai-ask-btn');
                    if (aiAskBtn) aiAskBtn.disabled = false;
                } catch (e) {
                    console.error('Failed to restore search results:', e);
                }
            }
        }
        
        // Load dark mode preference
        const darkMode = localStorage.getItem('chatwhiz_dark_mode');
        if (darkMode === 'false') {
            document.documentElement.classList.remove('dark');
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `glass-card rounded-lg p-4 max-w-sm animate-slide-up flex items-center space-x-3`;
        
        const icons = {
            success: 'fa-check-circle text-accent-success',
            error: 'fa-exclamation-triangle text-accent-danger',
            warning: 'fa-exclamation-circle text-accent-warning',
            info: 'fa-info-circle text-accent-info'
        };
        
        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 5000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatwhiz = new ChatWhizModern();
});
