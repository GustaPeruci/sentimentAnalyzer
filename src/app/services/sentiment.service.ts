import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject } from 'rxjs';

export interface SentimentResult {
  sentiment: string;
  sentiment_key: string;
  confidence: number;
  probabilities: { [key: string]: number };
  text: string;
  error?: string;
}

export interface AnalyticsData {
  sentiment: string;
  count: number;
  percentage: number;
  color: string;
}

export interface AnalyticsResponse {
  analytics: AnalyticsData[];
  recent_analyses: SentimentResult[];
  total_analyses: number;
  message?: string;
}

@Injectable({
  providedIn: 'root'
})
export class SentimentService {
  private apiUrl = '/api';
  private analysisHistory = new BehaviorSubject<SentimentResult[]>([]);
  public analysisHistory$ = this.analysisHistory.asObservable();

  constructor(private http: HttpClient) {}

  private getHttpHeaders(): HttpHeaders {
    return new HttpHeaders({
      'Content-Type': 'application/json'
    });
  }

  analyzeSentiment(text: string): Observable<SentimentResult> {
    const payload = { text: text };
    return this.http.post<SentimentResult>(
      `${this.apiUrl}/predict`, 
      payload, 
      { headers: this.getHttpHeaders() }
    );
  }

  batchAnalyze(texts: string[]): Observable<{ results: SentimentResult[], total_processed: number }> {
    const payload = { texts: texts };
    return this.http.post<{ results: SentimentResult[], total_processed: number }>(
      `${this.apiUrl}/batch-predict`, 
      payload, 
      { headers: this.getHttpHeaders() }
    );
  }

  getAnalytics(): Observable<AnalyticsResponse> {
    return this.http.get<AnalyticsResponse>(`${this.apiUrl}/analytics`);
  }

  getHistory(limit: number = 50): Observable<{ history: SentimentResult[], total_count: number }> {
    return this.http.get<{ history: SentimentResult[], total_count: number }>(
      `${this.apiUrl}/history?limit=${limit}`
    );
  }

  clearHistory(): Observable<{ message: string }> {
    return this.http.delete<{ message: string }>(`${this.apiUrl}/history`);
  }

  getModelInfo(): Observable<any> {
    return this.http.get(`${this.apiUrl}/model/info`);
  }

  trainModel(): Observable<any> {
    return this.http.post(`${this.apiUrl}/train`, {}, { headers: this.getHttpHeaders() });
  }

  // Local storage for analysis results
  addAnalysisResult(result: SentimentResult): void {
    const currentHistory = this.analysisHistory.value;
    const updatedHistory = [...currentHistory, result];
    
    // Keep only last 100 results to prevent memory issues
    if (updatedHistory.length > 100) {
      updatedHistory.splice(0, updatedHistory.length - 100);
    }
    
    this.analysisHistory.next(updatedHistory);
  }

  getLocalHistory(): SentimentResult[] {
    return this.analysisHistory.value;
  }

  clearLocalHistory(): void {
    this.analysisHistory.next([]);
  }

  // Helper method to get sentiment color
  getSentimentColor(sentiment: string): string {
    const colorMap: { [key: string]: string } = {
      'Tristeza': '#8E44AD',   // Purple
      'Alegria': '#F1C40F',    // Yellow
      'Raiva': '#E74C3C',      // Red
      'Surpresa': '#2ECC71'    // Green
    };
    return colorMap[sentiment] || '#95A5A6';
  }

  // Helper method to get sentiment label in Portuguese
  getSentimentLabel(sentiment: string): string {
    const labelMap: { [key: string]: string } = {
      'positive': 'Alegria',
      'negative': 'Tristeza',
      'neutral': 'Surpresa',
      'Alegria': 'Alegria',
      'Tristeza': 'Tristeza',
      'Surpresa': 'Surpresa'
    };
    return labelMap[sentiment] || sentiment;
  }

  // Academic functionality methods
  getAcademicAnalysis(): Observable<any> {
    return this.http.get(`${this.apiUrl}/academic-analysis`, {
      headers: this.getHttpHeaders()
    }).pipe(
      catchError(error => {
        console.error('Academic analysis failed:', error);
        return of({ error: 'Falha ao obter análise acadêmica' });
      })
    );
  }

  getPerformanceMetrics(): Observable<any> {
    return this.http.get(`${this.apiUrl}/performance-metrics`, {
      headers: this.getHttpHeaders()
    }).pipe(
      catchError(error => {
        console.error('Performance metrics failed:', error);
        return of({ error: 'Falha ao obter métricas de performance' });
      })
    );
  }

  getAcademicReport(): Observable<any> {
    return this.http.get(`${this.apiUrl}/academic-full-report`, {
      headers: this.getHttpHeaders()
    }).pipe(
      catchError(error => {
        console.error('Academic report failed:', error);
        return of({ error: 'Falha ao obter relatório acadêmico' });
      })
    );
  }

  getHealthStatus(): Observable<any> {
    return this.http.get('/health').pipe(
      catchError(error => {
        console.error('Health check failed:', error);
        return of({ status: 'unhealthy', error: error.message });
      })
    );
  }
}
