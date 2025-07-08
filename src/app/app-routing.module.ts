import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { MainPageComponent } from './components/main-page/main-page.component';
import { ResultsComponent } from './components/results/results.component';
import { AnalyticsComponent } from './components/analytics/analytics.component';

const routes: Routes = [
  { path: '', component: MainPageComponent },
  { path: 'resultados', component: ResultsComponent },
  { path: 'analise', component: AnalyticsComponent },
  { path: '**', redirectTo: '' } // Wildcard route for 404
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
