import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { MainPageComponent } from './components/main-page/main-page.component';
import { ResultsComponent } from './components/results/results.component';
import { AnalyticsComponent } from './components/analytics/analytics.component';

import { SentimentService } from './services/sentiment.service';

@NgModule({
  declarations: [
    AppComponent,
    MainPageComponent,
    ResultsComponent,
    AnalyticsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule
  ],
  providers: [SentimentService],
  bootstrap: [AppComponent]
})
export class AppModule { }
