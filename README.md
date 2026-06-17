# Expert Recommendation System

The **Expert Recommendation System** is an AI-powered internal directory and talent matching platform designed to connect users with the right internal subject matter experts based on explicit competency requirements or technical constraints.

By leveraging **Natural Language Processing (NLP)** techniques specifically **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** the system interprets unstructured user queries and matches them against a centralized database of expert skill descriptions.

---

## Core Key Features

### 1. Dynamic Expert Search (`Cari Ahli`)
* **Semantic Query Matching:** Users can input a real-world issue, problem description, or specific skill requirement (e.g., *"koneksi internet bermasalah"* / internet connection issues).
* **Ranked Recommendations:** The system processes the query, compares it against the expert database, and returns card-based recommendations sorted by a computed matching percentage score (filtering entries with a similarity index higher than `0.05`).
* **Interactive Profile Inspection:** Each card features a "Lihat Detail" action button that pops open a frontend modal overlay containing comprehensive profile telemetry, strict matching scores, and complete competency descriptions.

### 2. Centralized Database Management (`Kelola Pakar`)
* **Manual Expert Onboarding:** Administrators can add new experts manually by inputting their full name, official company email address, division/competency track (e.g., *IT Infrastructure, Legal & Hukum, Data Science & AI, Cyber Security*), and a granular narrative description of their professional expertise.
* **AI-Assisted Profile Generation:** Rather than typing manual copy, administrators can input short keywords (e.g., *wifi, error mac*), and the built-in **AI Description Generator** will run text similarity comparisons against existing descriptions or randomly assemble a professional fallback template profile on the fly.
* **Massive Bulk Import (`Import CSV Massal`):** Supports rapid data scaling via structural CSV templates. Administrators can download a pre-formatted template file (`template_pakar_baru.csv`), populate it with bulk records, and upload it via the **"Jalankan Import & Retrain AI"** capability, which executes batch insertion while bypassing duplicate records.
* **CRUD Database Operations:** An inline operational dashboard displays active profiles. Administrators can trigger interactive administrative controls for instant record updates (`Edit`) or permanent record removals (`Hapus`) featuring beautiful SweetAlert2 confirmation dialog overlays.

---

## Technical Implementation Details

* **Programming Ecosystem:** Written in **Python 3** using **Flask** as the primary backend server micro-framework.
* **Core Algorithms:** TF-IDF Vectorization for text tokenization and weighting, paired with Cosine Similarity metrics via `sklearn` to evaluate vector distances between user requests and database entries.
* **Database & Serializers:** SQLite for local relational data persistence, combined with `pickle` serialization to cache machine learning states.
* **User Interface:** Clean, reactive, and responsive frontend built using standard **HTML5**, **Tailwind CSS** components, and native asynchronous **JavaScript Fetch API** calls.

---

## Installation & Setup

```bash
# 1. Create a virtual environment to isolate the project's libraries
python -m venv .venv # Run this command if you are using Windows
python3 -m venv .venv # Run this command if you are using Linux/MacOS

# 2. Activate the created virtual environment based on your operating system
.venv\Scripts\activate # Run this command if you are using Windows
source .venv/bin/activate # Run this command if you are using Linux/MacOS

# 3. Install all the Python dependency libraries required by the system (flask, scikit-learn, pandas, etc.)
pip install -r requirements.txt