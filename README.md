# Premier League Predictions

Welcome to **Premier League Predictions**, a Django application designed to provide detailed insights and forecasts for the 2024/2025 Premier League matches. Optimized for desktop computers, this platform combines advanced data analysis, machine learning, and expert insights to deliver accurate match predictions, currently achieving an impressive accuracy rate of **66%**. 

## Technology

### Machine Learning Libraries
This application utilizes powerful libraries such as **NumPy**, **pandas**, and **scikit-learn** for data manipulation, analysis, and machine learning model implementation. These libraries are essential for processing large datasets, performing statistical analyses, and building predictive models.

### Database Management
Data is stored and managed using **PostgreSQL**, integrated via custom-built Django models, ensuring robust and scalable data handling.

### Caching
**Redis** caching is implemented to optimize performance and reduce response times.

### Web Scraping
**Selenium** and **BeautifulSoup** are employed for web scraping, ensuring real-time match data is collected.

### Automation and Real-Time Updates
An automated system for real-time match updates operates with **Celery**, enabling asynchronous task management and cron jobs for timely data updates.

### API Integration
Club information is gathered using the **Gemini Pro API**.

### Deployment
**Docker** and **Kubernetes (GKE)** are used for deploying the application, ensuring scalability, reliability, and efficient management of services.

## Prediction Process

### Data Collection
Comprehensive data is compiled from the Premier League's inception in the **1992/1993 season** to the **2024/2025 season**. The database includes detailed match statistics, ensuring a rich dataset for analysis.

### Data Validation
Collected data is validated to ensure consistency and accuracy, addressing any discrepancies before analysis.

### Feature Engineering
Detailed features for each match are created, focusing on:
- **Season and Matchweek Features**: General information, including environmental factors.
- **Last X Games Features**: Recent performance metrics like points, goals, and disciplinary actions, along with head-to-head records.
- **Season Points**: Tracking cumulative points for overall team performance.
- **Time Since Previous Match**: Elapsed time since each team's last match, considering player fatigue.
- **Cumulative Sum**: Maintaining cumulative statistics for home and away performances.

### Feature Difference Calculation
Differences between features for home and away teams are calculated, enhancing the model’s predictive capabilities.

### Data Preprocessing and Target Encoding
Essential preprocessing steps prepare the data for classifiers, converting the target feature into a numerical format suitable for machine learning.

### Training, Tuning, and Model Selection
Using **XGBClassifier**, **CatBoostClassifier**, and **LGBMClassifier**, models are trained and tuned. Their performance is evaluated using balanced accuracy scores to select the best model, which drives match predictions.

### Saving and Utilizing the Model
The preprocessed data and the best-performing model are saved for future predictions, ensuring accuracy for each match by repeating essential steps.

## Meet the Team
**Elior Nataf**, Software Developer based in Israel, specializing in backend development and machine learning. Currently part of **CitrusX**.

Visit the deployed website to see the predictions in action and for more details about how the process works: [Premier League Predictions](http://www.premier-league-predictions.xyz/).

Feel free to connect and chat via [LinkedIn](https://www.linkedin.com/in/elior/) or [email](mailto:eliorn23@gmail.com) for any inquiries, discussions, or collaborations.
