import { Component, OnInit, OnDestroy } from '@angular/core';
import { Router } from '@angular/router';
import { SentimentService, AnalyticsData } from '../../services/sentiment.service';
import { Chart, ChartConfiguration, ChartType } from 'chart.js';

// Register Chart.js components
import {
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

Chart.register(
  ArcElement,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

@Component({
  selector: 'app-analytics',
  templateUrl: './analytics.component.html',
  styleUrls: ['./analytics.component.css']
})
export class AnalyticsComponent implements OnInit, OnDestroy {
  analyticsData: AnalyticsData[] = [];
  totalAnalyses: number = 0;
  isLoading: boolean = false;
  errorMessage: string = '';
  
  // Chart instances
  donutChart: Chart | undefined;
  barChart: Chart | undefined;

  // Sample data matching the prototype
  sampleAnalyticsData: AnalyticsData[] = [
    { sentiment: 'Tristeza', count: 45, percentage: 45.5, color: '#8E44AD' },
    { sentiment: 'Surpresa', count: 27, percentage: 27.3, color: '#2ECC71' },
    { sentiment: 'Alegria', count: 18, percentage: 18.2, color: '#F1C40F' },
    { sentiment: 'Raiva', count: 9, percentage: 9.1, color: '#E74C3C' }
  ];

  constructor(
    private sentimentService: SentimentService,
    private router: Router
  ) {}

  ngOnInit(): void {
    this.loadAnalytics();
  }

  ngOnDestroy(): void {
    // Clean up chart instances
    if (this.donutChart) {
      this.donutChart.destroy();
    }
    if (this.barChart) {
      this.barChart.destroy();
    }
  }

  loadAnalytics(): void {
    this.isLoading = true;
    this.errorMessage = '';

    this.sentimentService.getAnalytics().subscribe({
      next: (response) => {
        if (response.analytics && response.analytics.length > 0) {
          this.analyticsData = response.analytics;
          this.totalAnalyses = response.total_analyses;
        } else {
          // Use sample data to match prototype
          this.analyticsData = this.sampleAnalyticsData;
          this.totalAnalyses = this.sampleAnalyticsData.reduce((sum, item) => sum + item.count, 0);
        }
        
        this.isLoading = false;
        
        // Create charts after data is loaded
        setTimeout(() => {
          this.createDonutChart();
          this.createBarChart();
        }, 100);
      },
      error: (error) => {
        console.warn('Could not load analytics from server, using sample data:', error);
        
        // Use sample data on error
        this.analyticsData = this.sampleAnalyticsData;
        this.totalAnalyses = this.sampleAnalyticsData.reduce((sum, item) => sum + item.count, 0);
        this.isLoading = false;
        
        // Create charts with sample data
        setTimeout(() => {
          this.createDonutChart();
          this.createBarChart();
        }, 100);
      }
    });
  }

  createDonutChart(): void {
    const canvas = document.getElementById('donutChart') as HTMLCanvasElement;
    if (!canvas) return;

    // Destroy existing chart
    if (this.donutChart) {
      this.donutChart.destroy();
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'doughnut' as ChartType,
      data: {
        labels: this.analyticsData.map(item => item.sentiment),
        datasets: [{
          data: this.analyticsData.map(item => item.percentage),
          backgroundColor: this.analyticsData.map(item => item.color),
          borderWidth: 0,
          cutout: '60%'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              usePointStyle: true,
              padding: 20,
              font: {
                size: 14,
                weight: '500'
              },
              generateLabels: (chart) => {
                const data = chart.data;
                if (data.labels && data.datasets.length) {
                  return data.labels.map((label, i) => {
                    const dataset = data.datasets[0];
                    const value = dataset.data[i] as number;
                    return {
                      text: `${label}: ${value.toFixed(1)}%`,
                      fillStyle: dataset.backgroundColor?.[i] as string,
                      strokeStyle: dataset.backgroundColor?.[i] as string,
                      lineWidth: 0,
                      pointStyle: 'circle'
                    };
                  });
                }
                return [];
              }
            }
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const sentiment = context.label;
                const percentage = context.parsed;
                const count = this.analyticsData.find(item => item.sentiment === sentiment)?.count || 0;
                return `${sentiment}: ${percentage.toFixed(1)}% (${count} análises)`;
              }
            }
          }
        }
      }
    };

    this.donutChart = new Chart(ctx, config);
  }

  createBarChart(): void {
    const canvas = document.getElementById('barChart') as HTMLCanvasElement;
    if (!canvas) return;

    // Destroy existing chart
    if (this.barChart) {
      this.barChart.destroy();
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const config: ChartConfiguration = {
      type: 'bar' as ChartType,
      data: {
        labels: this.analyticsData.map(item => item.sentiment),
        datasets: [{
          label: 'Quantidade de Análises',
          data: this.analyticsData.map(item => item.count),
          backgroundColor: this.analyticsData.map(item => item.color),
          borderRadius: 8,
          borderSkipped: false
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                const count = context.parsed.y;
                const sentiment = context.label;
                const item = this.analyticsData.find(d => d.sentiment === sentiment);
                const percentage = item?.percentage || 0;
                return `${sentiment}: ${count} análises (${percentage.toFixed(1)}%)`;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              stepSize: 5
            },
            grid: {
              color: 'rgba(0,0,0,0.1)'
            }
          },
          x: {
            grid: {
              display: false
            }
          }
        }
      }
    };

    this.barChart = new Chart(ctx, config);
  }

  refreshAnalytics(): void {
    this.loadAnalytics();
  }

  navigateToHome(): void {
    this.router.navigate(['/']);
  }

  navigateToResults(): void {
    this.router.navigate(['/resultados']);
  }

  getSentimentIcon(sentiment: string): string {
    const iconMap: { [key: string]: string } = {
      'Tristeza': 'fas fa-sad-tear',
      'Alegria': 'fas fa-smile',
      'Raiva': 'fas fa-angry',
      'Surpresa': 'fas fa-surprise'
    };
    return iconMap[sentiment] || 'fas fa-circle';
  }
}
