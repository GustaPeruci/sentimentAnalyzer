import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { SentimentService, SentimentResult } from '../../services/sentiment.service';

@Component({
  selector: 'app-main-page',
  templateUrl: './main-page.component.html',
  styleUrls: ['./main-page.component.css']
})
export class MainPageComponent implements OnInit {
  commentText: string = '';
  isAnalyzing: boolean = false;
  analysisResult: SentimentResult | null = null;
  errorMessage: string = '';

  constructor(
    private sentimentService: SentimentService,
    private router: Router
  ) {}

  ngOnInit(): void {
    // Check if model is loaded
    this.sentimentService.getModelInfo().subscribe({
      next: (info) => {
        if (!info.loaded) {
          console.warn('Model not loaded. You may need to train the model first.');
        }
      },
      error: (error) => {
        console.error('Error checking model status:', error);
      }
    });
  }

  onAnalyze(): void {
    if (!this.commentText.trim()) {
      this.errorMessage = 'Por favor, digite um comentário para analisar.';
      return;
    }

    this.isAnalyzing = true;
    this.errorMessage = '';
    this.analysisResult = null;

    this.sentimentService.analyzeSentiment(this.commentText.trim()).subscribe({
      next: (result) => {
        this.isAnalyzing = false;
        
        if (result.error) {
          this.errorMessage = result.error;
        } else {
          this.analysisResult = result;
          // Add to local history
          this.sentimentService.addAnalysisResult(result);
          
          // Navigate to results page after a short delay to show the result
          setTimeout(() => {
            this.router.navigate(['/resultados']);
          }, 2000);
        }
      },
      error: (error) => {
        this.isAnalyzing = false;
        console.error('Analysis error:', error);
        this.errorMessage = 'Erro ao analisar o sentimento. Tente novamente.';
        
        if (error.status === 0) {
          this.errorMessage = 'Erro de conexão. Verifique se o servidor está funcionando.';
        } else if (error.status === 500) {
          this.errorMessage = 'Erro interno do servidor. Pode ser necessário treinar o modelo primeiro.';
        }
      }
    });
  }

  onKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.onAnalyze();
    }
  }

  getSentimentColor(sentiment: string): string {
    return this.sentimentService.getSentimentColor(sentiment);
  }

  formatConfidence(confidence: number): string {
    return (confidence * 100).toFixed(1);
  }

  navigateToResults(): void {
    this.router.navigate(['/resultados']);
  }

  navigateToAnalytics(): void {
    this.router.navigate(['/analise']);
  }
}
