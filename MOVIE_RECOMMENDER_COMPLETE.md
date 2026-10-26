# 🎬 سیستم توصیه‌گر فیلم: طراحی + توسعه + پیاده‌سازی

## 📌 فهرست

1. **مرحله ۱: مسئله و حل**
2. **مرحله ۲: طراحی (Wireframe + Flowchart)**
3. **مرحله ۳: PRD (Product Requirements Document)**
4. **مرحله ۴: Jira Tasks**
5. **مرحله ۵: نقشه‌ی راهی کدنویسی**
6. **مرحله ۶: کد مرحله‌به‌مرحله**
7. **مرحله ۷: پیاده‌سازی و Deploy**

---

# 🎯 مرحله ۱: مسئله و حل

## مسئله

```
🎬 پلتفرم تماشای فیلم (مثل نتفلیکس)
  ↓
👤 کاربر جدید وارد میشود
  ↓
😕 ۵۰۰۰+ فیلم وجود دارد
  ↓
❌ کاربر نمی‌دونه کدوم فیلم دیده بشه
  ↓
📉 تماشای کاهش می‌یابد
```

---

## حل

```
🎬 سیستم توصیه‌گر هوشمند
  ↓
📊 تحلیل: کاربر چه فیلم‌هایی دیده؟
  ↓
🧠 مدل ML: الگوهای تماشا پیدا کن
  ↓
✨ پیشنهاد: ۱۰ فیلم برتر برای این کاربر
  ↓
📈 نتیجه: تماشا ۳۰% افزایش می‌یابد
```

---

# 🎨 مرحله ۲: طراحی

## ۲.۱ Flowchart (نقشه‌ی منطقی)

### چطور بکشی؟

**ابزار:** Excalidraw.com (رایگان، سریع)

```
مراحل:
1. برو به https://excalidraw.com
2. یک flowchart بکش (نمونه زیر)
3. Screenshot گرفت یا Ctrl+S کن
```

### Flowchart نمونه:

```
┌─────────────────────────────────────────────────────┐
│         🎬 سیستم توصیه‌گر فیلم                    │
└─────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   کاربر وارد صفحه می‌شود      │
        │   (user_id = 105)            │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  سابقه‌ی تماشای کاربر را      │
        │  از دیتابیس بیرون کن          │
        │  (۳۰ فیلمی که دیده)         │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  ویژگی‌های کاربر محاسبه کن:  │
        │  • میانگین امتیاز            │
        │  • ژانرهای مورد علاقه        │
        │  • بازیگرهای مورد علاقه      │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  تمام فیلم‌های ندیده را      │
        │  صاف کن (۴۹۷۰ فیلم باقی)   │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  برای هر فیلم ویژگی‌ها       │
        │  محاسبه کن:                  │
        │  • ژانر                      │
        │  • بازیگرها                  │
        │  • کارگردان                  │
        │  • امتیاز IMDb              │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  مدل ML پیش‌بینی کند:       │
        │  "احتمال: ۸۷%"               │
        │  (کاربر این فیلم دوست خواهد)│
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  فیلم‌ها را بر اساس          │
        │  احتمال ترتیب بده            │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  ۱۰ فیلم برتر را به          │
        │  کاربر نشان بده              │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  کاربر روی توصیه کلیک کند     │
        │  یا نه                       │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │  نتایج را ثبت کن             │
        │  (برای بهتر شدن مدل)        │
        └───────────────────────────────┘
```

---

## ۲.۲ Wireframe (طرح صفحه)

### چطور بکشی؟

**ابزار:** Figma.com (رایگان) یا Wireframe.cc

### Wireframe نمونه:

```
┌──────────────────────────────────────────────────┐
│  🎬 NetFilm - صفحه‌ی توصیه‌های شخصی          │
├──────────────────────────────────────────────────┤
│                                                  │
│  👤 سلام محمد! 👋                               │
│                                                  │
│  ✨ فیلم‌های انتخاب‌شده برای تو:              │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │ 🎥 [عکس]  فیلم ۱                       │  │
│  │           نام: Interstellar            │  │
│  │           ژانر: علمی-تخیلی             │  │
│  │           امتیاز: ⭐⭐⭐⭐⭐ (4.8)    │  │
│  │           احتمال پسند: 92%            │  │
│  │           [تماشا کن] [نه مرسی]       │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │ 🎥 [عکس]  فیلم ۲                       │  │
│  │           نام: The Matrix             │  │
│  │           ژانر: علمی-تخیلی             │  │
│  │           امتیاز: ⭐⭐⭐⭐⭐ (4.7)    │  │
│  │           احتمال پسند: 89%            │  │
│  │           [تماشا کن] [نه مرسی]       │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ... (۸ فیلم دیگر)                             │
│                                                  │
│  [صفحه‌ی قبل] • ۱/۱۰ • [صفحه‌ی بعد]          │
└──────────────────────────────────────────────────┘
```

---

# 📄 مرحله ۳: PRD (Product Requirements Document)

## PRD کامل

```markdown
# Product Requirements Document (PRD)
## سیستم توصیه‌گر فیلم (Movie Recommender System)

---

## ۱. خلاصه‌ی محصول

سیستمی برای توصیه‌ی فیلم‌های شخصی‌سازی‌شده به کاربران بر اساس:
- سابقه‌ی تماشای کاربر
- تشابه فیلم‌ها (ژانر، بازیگر، کارگردان)
- رفتار کاربرهای مشابه

---

## ۲. مسئله و فرصت

### مسئله
- کاربران جدید گم‌گشته میان ۵۰۰۰+ فیلم
- نرخ تبدیل (Conversion) پایین: ۳%
- کاربران چند دقیقه بیشتر تلاش نمی‌کنند

### فرصت
- هر توصیه صحیح = ۱۵ دقیقه تماشا اضافی
- ۳۰% افزایش تماشا = ۳۰% افزایش درآمد

---

## ۳. کاربرهای تارگت

| کاربر | شرح |
|---|---|
| **جدید** | اولین بار وارد شده، ۰ تماشا |
| **فعال** | ماهانه ۵۰+ ساعت تماشا |
| **غیرفعال** | ۳ ماه بدون تماشا |

---

## ۴. هدف‌های کلیدی (KPIs)

```
۱. Recommendation Click-Through Rate (CTR)
   فعلی: ۲% → هدف: ۱۵%

۲. Engagement (مدت زمان تماشا)
   فعلی: ۲۰ دقیقه/روز → هدف: ۲۶ دقیقه/روز

۳. Conversion (تماشا/توصیه)
   فعلی: ۱۲% → هدف: ۲۵%

۴. User Retention (ماندگاری)
   فعلی: ۴۰% ماهیانه → هدف: ۶۰%
```

---

## ۵. ویژگی‌های اصلی

### ویژگی ۱: نمایش توصیه‌ها
```
• ۱۰ فیلم پیشنهادی
• برای هر فیلم:
  - عکس (poster)
  - نام
  - ژانر
  - امتیاز (IMDb)
  - احتمال پسند (۰-۱۰۰%)
  - دکمه‌ی "تماشا کن"
```

### ویژگی ۲: دلیل توصیه
```
"چرا این فیلم برای تو؟"
  • "تو از فیلم‌های sci-fi خوشت می‌آد"
  • "۱۰۰ نفر مثل تو این رو دوست داشتند"
  • "بازیگر مورد علاقه‌ات: Tom Hanks"
```

### ویژگی ۳: فیلتر و مرتب‌سازی
```
فیلتر:
  • ژانر
  • سال انتشار
  • امتیاز حداقل

مرتب‌سازی:
  • احتمال پسند (default)
  • امتیاز IMDb
  • جدید‌ترین
```

### ویژگی ۴: بازخورد
```
کاربر می‌تونه:
  • "دوست داشتم" ❤️
  • "دوست نداشتم" 👎
  • "بعداً"
  
→ مدل یاد می‌گیره
```

---

## ۶. تکنولوژی‌های مورد استفاده

```
Frontend:
  • React.js
  • Material-UI (طراحی)

Backend:
  • Python + Flask
  • PostgreSQL (دیتابیس)

ML Model:
  • scikit-learn
  • Pandas + NumPy
  
DevOps:
  • Docker
  • AWS (یا Heroku)
```

---

## ۷. معماری سیستم

```
                    ┌──────────────┐
                    │  Frontend    │
                    │  (React)     │
                    └──────┬───────┘
                           │
                           ↓
                    ┌──────────────┐
                    │  API Server  │
                    │  (Flask)     │
                    └──────┬───────┘
                           │
                ┌──────────┼──────────┐
                ↓          ↓          ↓
          ┌─────────┐ ┌────────┐ ┌──────────┐
          │  User   │ │ Movie  │ │   ML     │
          │   DB    │ │  DB    │ │  Model   │
          │(Postgres)│ │(Postgres)│(pickle) │
          └─────────┘ └────────┘ └──────────┘
```

---

## ۸. فاز توسعه (Timeline)

### فاز ۱: MVP (Minimum Viable Product) - ۴ هفته

```
🔵 هفته ۱: Data Preparation
  • دیتا جمع‌آوری و تمیزی

🟡 هفته ۲: Model Development
  • ساخت مدل ML

🟠 هفته ۳: Backend
  • API endpoints

🔴 هفته ۴: Frontend + Test
  • صفحه‌ی UI
  • تست کامل
```

### فاز ۲: بهبود - ۲ هفته

```
• Personalization بهتر
• Performance optimization
• Analytics
```

---

## ۹. معیارهای موفقیت

```
✅ Technical:
   • Response time < ۲ ثانیه
   • Model accuracy > ۸۵%
   • ۹۹.۹% uptime

✅ Business:
   • CTR > ۱۵%
   • Retention > ۶۰%
   • درآمد ۲۰% افزایش
```

---

## ۱۰. خطرات و راهکارها

| خطر | احتمال | راهکار |
|---|---|---|
| مدل نادرست | بالا | Cross-validation دقیق |
| سرور کند | متوسط | Caching + CDN |
| کاربرها نپذیرند | پایین | A/B testing |

---

## ۱۱. Next Steps

```
۱. تایید PRD توسط stakeholders
۲. Jira tasks ایجاد
۳. Sprint planning
۴. شروع توسعه
```
```

---

# 📋 مرحله ۴: Jira Tasks

## نحوه‌ی ساختار در Jira

```
EPIC: Movie Recommender System (20 weeks)
│
├─ STORY 1: Data Preparation (Week 1)
│  ├─ TASK 1.1: دیتای فیلم‌ها جمع‌آوری [3 ساعت]
│  ├─ TASK 1.2: دیتای rating‌ها جمع‌آوری [3 ساعت]
│  ├─ TASK 1.3: Data cleaning و preprocessing [8 ساعت]
│  └─ TASK 1.4: Exploratory Data Analysis (EDA) [6 ساعت]
│
├─ STORY 2: Model Development (Week 2-3)
│  ├─ TASK 2.1: Feature Engineering [16 ساعت]
│  ├─ TASK 2.2: مدل Content-based [12 ساعت]
│  ├─ TASK 2.3: مدل Collaborative Filtering [16 ساعت]
│  ├─ TASK 2.4: مدل Ensemble [12 ساعت]
│  ├─ TASK 2.5: Hyperparameter tuning [12 ساعت]
│  └─ TASK 2.6: Model evaluation و validation [8 ساعت]
│
├─ STORY 3: Backend API (Week 4)
│  ├─ TASK 3.1: Flask project setup [4 ساعت]
│  ├─ TASK 3.2: Database schema design [6 ساعت]
│  ├─ TASK 3.3: User management API [8 ساعت]
│  ├─ TASK 3.4: /recommend endpoint [8 ساعت]
│  ├─ TASK 3.5: Authentication (JWT) [6 ساعت]
│  └─ TASK 3.6: API testing [6 ساعت]
│
├─ STORY 4: Frontend UI (Week 5)
│  ├─ TASK 4.1: React setup و routing [4 ساعت]
│  ├─ TASK 4.2: صفحه‌ی توصیه‌ها UI [8 ساعت]
│  ├─ TASK 4.3: Movie detail page [6 ساعت]
│  ├─ TASK 4.4: Integration با API [6 ساعت]
│  └─ TASK 4.5: UI testing [4 ساعت]
│
├─ STORY 5: Analytics و Monitoring (Week 6)
│  ├─ TASK 5.1: Click tracking [4 ساعت]
│  ├─ TASK 5.2: Dashboard [6 ساعت]
│  └─ TASK 5.3: Logging و error handling [4 ساعت]
│
└─ STORY 6: Deployment (Week 7)
   ├─ TASK 6.1: Docker setup [4 ساعت]
   ├─ TASK 6.2: AWS/Heroku config [4 ساعت]
   ├─ TASK 6.3: CI/CD pipeline [6 ساعت]
   └─ TASK 6.4: Production monitoring [4 ساعت]
```

---

# 🔥 مرحله ۵: نقشه‌ی راهی کدنویسی

## فایل‌های پروژه

```
movie_recommender/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── user_profiles.csv
│
├── notebooks/
│   ├── 01_eda.ipynb                (EDA)
│   ├── 02_feature_engineering.ipynb (ویژگی‌ها)
│   └── 03_model_training.ipynb      (مدل)
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              (بارگذاری داده)
│   ├── feature_engineer.py          (ویژگی‌ها)
│   ├── model.py                     (مدل ML)
│   ├── recommender.py               (توصیه‌گر)
│   └── utils.py                     (کمکی)
│
├── backend/
│   ├── app.py                       (Flask)
│   ├── models.py                    (DB models)
│   ├── routes.py                    (API endpoints)
│   └── config.py                    (تنظیمات)
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── RecommendationList.jsx
│   │   │   ├── MovieCard.jsx
│   │   │   └── Navbar.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Recommendations.jsx
│   │   │   └── MovieDetail.jsx
│   │   ├── App.jsx
│   │   └── index.js
│   └── package.json
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_engineer.py
│   ├── test_model.py
│   └── test_api.py
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
│
├── requirements.txt                 (Python packages)
├── .gitignore
├── README.md
└── setup.py
```

---

# 💻 مرحله ۶: کد مرحله‌به‌مرحله

## قسمت ۱: بارگذاری و تحلیل داده (۲ روز)

### فایل: `src/data_loader.py`

```python
"""
داده‌های فیلم را بارگذاری و تمیز کن
"""
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_data(self):
        """تمام داده‌ها را بارگذاری کن"""
        print("🔵 بارگذاری داده‌ها...")
        
        self.movies = pd.read_csv(self.data_dir / "movies.csv")
        self.ratings = pd.read_csv(self.data_dir / "ratings.csv")
        self.users = pd.read_csv(self.data_dir / "user_profiles.csv")
        
        print(f"   ✅ فیلم‌ها: {len(self.movies)}")
        print(f"   ✅ rating‌ها: {len(self.ratings)}")
        print(f"   ✅ کاربرها: {len(self.users)}")
        
        return self.movies, self.ratings, self.users
    
    def clean_data(self):
        """داده‌ها را تمیز کن"""
        print("\n🟡 تمیزی داده‌ها...")
        
        # NaN ها رو حذف کن
        self.ratings = self.ratings.dropna()
        self.movies = self.movies.dropna(subset=['title', 'genre'])
        
        # duplicates رو حذف کن
        self.ratings = self.ratings.drop_duplicates()
        
        print("   ✅ تمیز شد")
        
        return self.ratings, self.movies
    
    def get_stats(self):
        """آمار داده‌ها"""
        stats = {
            'total_users': self.ratings['user_id'].nunique(),
            'total_movies': self.ratings['movie_id'].nunique(),
            'total_ratings': len(self.ratings),
            'avg_rating': self.ratings['rating'].mean(),
            'date_range': f"{self.ratings['timestamp'].min()} to {self.ratings['timestamp'].max()}"
        }
        
        print("\n📊 آمار:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return stats


# استفاده
if __name__ == "__main__":
    loader = DataLoader()
    movies, ratings, users = loader.load_data()
    ratings, movies = loader.clean_data()
    stats = loader.get_stats()
```

---

## قسمت ۲: ویژگی‌ها (۳ روز)

### فایل: `src/feature_engineer.py`

```python
"""
ویژگی‌های کاربر و فیلم را محاسبه کن
"""
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, ratings, movies, users):
        self.ratings = ratings
        self.movies = movies
        self.users = users
    
    def user_features(self):
        """ویژگی‌های کاربر"""
        print("🔵 ویژگی‌های کاربر...")
        
        user_stats = self.ratings.groupby('user_id').agg({
            'rating': ['mean', 'count', 'std'],
            'movie_id': 'nunique'
        }).round(2)
        
        user_stats.columns = [
            'mean_rating',
            'total_ratings',
            'rating_std',
            'unique_movies'
        ]
        
        print(f"   ✅ {len(user_stats)} کاربر")
        return user_stats
    
    def movie_features(self):
        """ویژگی‌های فیلم"""
        print("🟡 ویژگی‌های فیلم...")
        
        movie_stats = self.ratings.groupby('movie_id').agg({
            'rating': ['mean', 'count'],
            'user_id': 'nunique'
        }).round(2)
        
        movie_stats.columns = [
            'mean_rating',
            'rating_count',
            'unique_users'
        ]
        
        # نرخ لایک
        likes = (self.ratings[self.ratings['rating'] > 3.5]
                .groupby('movie_id')
                .size())
        
        movie_stats['like_ratio'] = (
            likes / movie_stats['rating_count']
        ).fillna(0).round(2)
        
        print(f"   ✅ {len(movie_stats)} فیلم")
        return movie_stats
    
    def similarity_matrix(self):
        """شباهت بین فیلم‌ها"""
        print("🟠 ماتریس شباهت...")
        
        # User-item matrix
        user_item = self.ratings.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating',
            fill_value=0
        )
        
        # شباهت بین فیلم‌ها (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity(user_item.T)
        similarity_df = pd.DataFrame(
            similarity,
            index=user_item.columns,
            columns=user_item.columns
        )
        
        print(f"   ✅ ماتریس {similarity_df.shape}")
        return similarity_df


# استفاده
if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader()
    movies, ratings, users = loader.load_data()
    ratings, movies = loader.clean_data()
    
    fe = FeatureEngineer(ratings, movies, users)
    user_feat = fe.user_features()
    movie_feat = fe.movie_features()
    similarity = fe.similarity_matrix()
```

---

## قسمت ۳: مدل (۴ روز)

### فایل: `src/model.py`

```python
"""
مدل توصیه‌گر
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle

class RecommenderModel:
    def __init__(self, ratings, movies, movie_features):
        self.ratings = ratings
        self.movies = movies
        self.movie_features = movie_features
        self.model = None
    
    def prepare_training_data(self):
        """داده‌های آموزشی را آماده کن"""
        print("🔵 آماده‌سازی داده‌های آموزشی...")
        
        # برچسب: فیلم لایک شده یا نه؟
        self.ratings['liked'] = (self.ratings['rating'] > 3.5).astype(int)
        
        # ویژگی‌های کاربر + فیلم
        X = []
        y = []
        
        for _, row in self.ratings.iterrows():
            user_id = row['user_id']
            movie_id = row['movie_id']
            liked = row['liked']
            
            # ویژگی‌های کاربر
            user_mean = self.ratings[
                self.ratings['user_id'] == user_id
            ]['rating'].mean()
            
            # ویژگی‌های فیلم
            movie_mean = self.movie_features.loc[
                movie_id, 'mean_rating'
            ] if movie_id in self.movie_features.index else 0
            
            movie_like = self.movie_features.loc[
                movie_id, 'like_ratio'
            ] if movie_id in self.movie_features.index else 0
            
            X.append([user_mean, movie_mean, movie_like])
            y.append(liked)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"   ✅ داده: {X.shape}")
        return X, y
    
    def train(self):
        """مدل را آموزش بده"""
        print("\n🟡 آموزش مدل...")
        
        X, y = self.prepare_training_data()
        
        # تقسیم
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # مدل
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # ارزیابی
        train_acc = accuracy_score(
            y_train,
            self.model.predict(X_train)
        )
        test_acc = accuracy_score(
            y_test,
            self.model.predict(X_test)
        )
        
        print(f"   Train Accuracy: {train_acc:.2f}")
        print(f"   Test Accuracy: {test_acc:.2f}")
        
        return self.model
    
    def save_model(self, path: str = "models/recommender.pkl"):
        """مدل را ذخیره کن"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"   ✅ مدل ذخیره شد: {path}")
    
    def load_model(self, path: str = "models/recommender.pkl"):
        """مدل را لود کن"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"   ✅ مدل لود شد: {path}")


# استفاده
if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    
    loader = DataLoader()
    movies, ratings, users = loader.load_data()
    ratings, movies = loader.clean_data()
    
    fe = FeatureEngineer(ratings, movies, users)
    movie_feat = fe.movie_features()
    
    model = RecommenderModel(ratings, movies, movie_feat)
    model.train()
    model.save_model()
```

---

## قسمت ۴: توصیه‌گر (۲ روز)

### فایل: `src/recommender.py`

```python
"""
توصیه‌های نهایی برای کاربر
"""
import pandas as pd
import numpy as np

class Recommender:
    def __init__(self, model, ratings, movies, movie_features):
        self.model = model
        self.ratings = ratings
        self.movies = movies
        self.movie_features = movie_features
    
    def recommend(self, user_id: int, top_n: int = 10):
        """
        کاربر را توصیه بده
        
        Args:
            user_id: شناسه‌ی کاربر
            top_n: تعداد توصیه
        
        Returns:
            list: فیلم‌های پیشنهادی
        """
        print(f"🎬 توصیه برای کاربر {user_id}...")
        
        # فیلم‌های ندیده
        watched = set(
            self.ratings[
                self.ratings['user_id'] == user_id
            ]['movie_id'].values
        )
        
        candidates = self.movies[
            ~self.movies['movie_id'].isin(watched)
        ].copy()
        
        if len(candidates) == 0:
            print("   ❌ کاربر تمام فیلم‌ها را دیده")
            return []
        
        # پیش‌بینی برای هر فیلم
        recommendations = []
        
        user_mean = self.ratings[
            self.ratings['user_id'] == user_id
        ]['rating'].mean()
        
        for _, movie in candidates.iterrows():
            movie_id = movie['movie_id']
            
            movie_mean = self.movie_features.loc[
                movie_id, 'mean_rating'
            ] if movie_id in self.movie_features.index else 0
            
            movie_like = self.movie_features.loc[
                movie_id, 'like_ratio'
            ] if movie_id in self.movie_features.index else 0
            
            # پیش‌بینی احتمال
            X = np.array([[user_mean, movie_mean, movie_like]])
            prob = self.model.predict_proba(X)[0][1]
            
            recommendations.append({
                'movie_id': movie_id,
                'title': movie['title'],
                'genre': movie.get('genre', 'Unknown'),
                'rating': movie.get('rating', 0),
                'probability': round(prob * 100, 2)
            })
        
        # ترتیب بر اساس احتمال
        recommendations = sorted(
            recommendations,
            key=lambda x: x['probability'],
            reverse=True
        )[:top_n]
        
        # نمایش
        print(f"\n✨ {len(recommendations)} فیلم پیشنهادی:\n")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   ژانر: {rec['genre']}")
            print(f"   احتمال: {rec['probability']}%")
            print()
        
        return recommendations


# استفاده
if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineer import FeatureEngineer
    from model import RecommenderModel
    
    loader = DataLoader()
    movies, ratings, users = loader.load_data()
    ratings, movies = loader.clean_data()
    
    fe = FeatureEngineer(ratings, movies, users)
    movie_feat = fe.movie_features()
    
    rm = RecommenderModel(ratings, movies, movie_feat)
    rm.train()
    
    rec = Recommender(rm.model, ratings, movies, movie_feat)
    recommendations = rec.recommend(user_id=1, top_n=10)
```

---

## قسمت ۵: Flask API (۲ روز)

### فایل: `backend/app.py`

```python
"""
Flask API برای سیستم توصیه
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.insert(0, '../src')

from recommender import Recommender
from model import RecommenderModel
from data_loader import DataLoader
from feature_engineer import FeatureEngineer

app = Flask(__name__)
CORS(app)

# بارگذاری مدل
print("🔵 بارگذاری سیستم...")
loader = DataLoader()
movies, ratings, users = loader.load_data()
ratings, movies = loader.clean_data()

fe = FeatureEngineer(ratings, movies, users)
movie_feat = fe.movie_features()

rm = RecommenderModel(ratings, movies, movie_feat)
rm.train()

recommender = Recommender(rm.model, ratings, movies, movie_feat)

print("✅ سیستم آماده است!")

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """دریافت توصیه‌ها"""
    try:
        data = request.json
        user_id = data.get('user_id')
        top_n = data.get('top_n', 10)
        
        if not user_id:
            return jsonify({'error': 'user_id required'}), 400
        
        recommendations = recommender.recommend(user_id, top_n)
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """بررسی سلامتی سرور"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

# 🚀 مرحله ۷: پیاده‌سازی و Deploy

## ۷.۱ Docker

### فایل: `docker/Dockerfile.backend`

```dockerfile
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY backend/ .
COPY src/ ../src/

EXPOSE 5000

CMD ["python", "app.py"]
```

---

## ۷.۲ Deployment

### Deploy به Heroku

```bash
# ۱. Heroku CLI نصب کن
curl https://cli-assets.heroku.com/install.sh | sh

# ۲. وارد شو
heroku login

# ۳. app ایجاد کن
heroku create movie-recommender-app

# ۴. push کن
git push heroku main

# ۵. مشاهده logs
heroku logs --tail
```

---

# 📝 خلاصه

```
✅ مرحله ۱: مسئله و حل (۱ روز)
✅ مرحله ۲: طراحی (۲ روز)
   • Flowchart (Draw.io)
   • Wireframe (Figma)
✅ مرحله ۳: PRD (۱ روز)
✅ مرحله ۴: Jira Tasks (۱ روز)
✅ مرحله ۵: نقشه‌ی راهی (۱ روز)
✅ مرحله ۶: کدنویسی (۲ هفته)
   • قسمت ۱: داده (۲ روز)
   • قسمت ۲: ویژگی (۳ روز)
   • قسمت ۳: مدل (۴ روز)
   • قسمت ۴: توصیه (۲ روز)
   • قسمت ۵: API (۲ روز)
✅ مرحله ۷: پیاده‌سازی (۳ روز)

🎉 مجموعی: ۳ هفته ✨
```

---

**حالا می‌تونی خودت از صفر شروع کنی و توصیه‌گر خودت رو بسازی! 🚀**

