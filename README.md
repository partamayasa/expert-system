# Expert Recommendation System

The **Expert Recommendation System** is an AI-powered internal directory and talent matching platform designed to connect users with the right internal subject matter experts based on explicit competency requirements or technical constraints.

By leveraging **Natural Language Processing (NLP)** techniques specifically **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** the system interprets unstructured user queries and matches them against a centralized database of expert skill descriptions.

---

## Core Key Features

### 1. Dynamic Expert Search (`Cari Ahli`)
* **Semantic Query Matching:** Users can input a real-world issue, problem description, or solution keywords. The system compares the query against the combined text of **Problem** and **Solution** from the historical problem-solving database.
* **Ranked Recommendations (Card Riwayat Masalah):** The system processes the query and returns card-based recommendations sorted by matching score (filtering entries with a similarity index higher than `0.05`). Each card displays the matched issue ("Kendala Serupa") and the resolution ("Solusi Yang Diterapkan").
* **Interactive Profile & History Inspection:** Each card features a "Lihat Profil & Riwayat" button that pops open a frontend modal overlay showing the expert's full profile details (Name, Division, Email, and general "Keahlian Umum" description) along with a complete history of all issues they have resolved.

### 2. Centralized Database Management (`Kelola Pakar`)
* **Manual Expert Onboarding:** Administrators can add new experts manually by inputting their full name, official company email address, division/competency track (e.g., *Teknologi, Administrasi, Manajemen Risiko, Human Capital, Keuangan & Pajak, Legal & Hukum, Pemasaran Digital, Operasional Pabrik*), and a narrative description of their professional expertise.
* **Massive Bulk Import (`Import CSV Massal`):** Supports rapid data scaling via structural CSV templates. Administrators can download a pre-formatted template file (`template_pakar_baru.csv`) containing 1 clean Indonesian example row (featuring valid JSON format for problems and solutions).
* **CRUD Database Operations & UI Enhancements:** An inline operational dashboard displays active profiles. It features **real-time client-side search** and **pagination (10 items per page)** for the active expert database table, alongside interactive controls for instant record updates (`Edit`) or permanent removals (`Hapus`).

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