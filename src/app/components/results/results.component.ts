import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { SentimentService, SentimentResult } from '../../services/sentiment.service';

@Component({
  selector: 'app-results',
  templateUrl: './results.component.html',
  styleUrls: ['./results.component.css']
})
export class ResultsComponent implements OnInit {
  analysisHistory: SentimentResult[] = [];
  isLoading: boolean = false;
  errorMessage: string = '';

  // Sample Portuguese texts matching the prototype
  sampleAnalyses: SentimentResult[] = [
    {
      sentiment: 'Tristeza',
      sentiment_key: 'tristeza',
      confidence: 0.85,
      probabilities: { tristeza: 0.85, alegria: 0.10, surpresa: 0.05 },
      text: 'A entrega está 3 dias atrasada...'
    },
    {
      sentiment: 'Alegria',
      sentiment_key: 'alegria',
      confidence: 0.92,
      probabilities: { alegria: 0.92, surpresa: 0.06, tristeza: 0.02 },
      text: 'Ótimo produto, superou minhas expectativas!'
    },
    {
      sentiment: 'Raiva',
      sentiment_key: 'raiva',
      confidence: 0.78,
      probabilities: { raiva: 0.78, tristeza: 0.15, surpresa: 0.07 },
      text: 'Esse produto parou de funcionar depois de uma semana'
    },
    {
      sentiment: 'Surpresa',
      sentiment_key: 'surpresa',
      confidence: 0.71,
      probabilities: { surpresa: 0.71, alegria: 0.20, tristeza: 0.09 },
      text: 'Eu não esperava essa funcionalidade...'
    }
  ];

  constructor(
    private sentimentService: SentimentService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadAnalysisHistory();
  }

  loadAnalysisHistory(): void {
    this.isLoading = true;
    this.errorMessage = '';

    // Get local history first
    const localHistory = this.sentimentService.getLocalHistory();
    
    if (localHistory.length > 0) {
      this.analysisHistory = [...localHistory];
      this.isLoading = false;
    } else {
      // If no local history, show sample analyses to match prototype
      this.analysisHistory = [...this.sampleAnalyses];
      this.isLoading = false;
    }

    // Also try to get server history
    this.sentimentService.getHistory(20).subscribe({
      next: (response) => {
        if (response.history && response.history.length > 0) {
          // Merge server history with local history, avoiding duplicates
          const combinedHistory = [...localHistory, ...response.history];
          const uniqueHistory = this.removeDuplicates(combinedHistory);
          this.analysisHistory = uniqueHistory.slice(-20); // Keep last 20
        }
        this.isLoading = false;
      },
      error: (error) => {
        console.warn('Could not load server history:', error);
        // Keep showing local history or samples
        this.isLoading = false;
      }
    });
  }

  private removeDuplicates(analyses: SentimentResult[]): SentimentResult[] {
    const seen = new Set();
    return analyses.filter(analysis => {
      const key = `${analysis.text}-${analysis.sentiment}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  getSentimentColor(sentiment: string): string {
    return this.sentimentService.getSentimentColor(sentiment);
  }

  formatConfidence(confidence: number): string {
    return (confidence * 100).toFixed(1);
  }

  getConfidenceWidth(confidence: number): string {
    return (confidence * 100).toFixed(1) + '%';
  }

  clearHistory(): void {
    if (confirm('Tem certeza que deseja limpar o histórico de análises?')) {
      this.sentimentService.clearLocalHistory();
      
      // Also clear server history
      this.sentimentService.clearHistory().subscribe({
        next: () => {
          this.analysisHistory = [];
          console.log('History cleared successfully');
        },
        error: (error) => {
          console.warn('Could not clear server history:', error);
          // Still clear local history
          this.analysisHistory = [];
        }
      });
    }
  }

  navigateToHome(): void {
    this.router.navigate(['/']);
  }

  navigateToAnalytics(): void {
    this.router.navigate(['/analise']);
  }

  refreshHistory(): void {
    this.loadAnalytics();
  }

  getSentimentDisplayName(sentiment: string): string {
    const displayNames: { [key: string]: string } = {
      'tristeza': 'Tristeza',
      'alegria': 'Alegria', 
      'raiva': 'Raiva',
      'surpresa': 'Surpresa',
      'Tristeza': 'Tristeza',
      'Alegria': 'Alegria',
      'Raiva': 'Raiva', 
      'Surpresa': 'Surpresa'
    };
    return displayNames[sentiment] || sentiment;
  }

  // Helper method to convert object to key-value pairs for template iteration
  objectEntries(obj: { [key: string]: number }): { key: string, value: number }[] {
    return Object.entries(obj).map(([key, value]) => ({ key, value }));
  }

  private loadAnalytics(): void {
    this.loadAnalysisHistory();
  }
}
