import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import mysql.connector
from sqlalchemy import create_engine
from datetime import datetime

# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="warehouse"
    )

# Create SQLAlchemy engine
engine = create_engine('mysql+mysqlconnector://root:@localhost/warehouse')

# Set page config
st.set_page_config(
    page_title="Warehouse Management System",
    page_icon="📦",
    layout="wide"
)

# =============================================
# ID Generation Functions
# =============================================

def generate_material_id():
    """Generate material ID using WZ + year + 4-digit sequence"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT CONCAT('WZ', DATE_FORMAT(NOW(), '%Y'), 
                     LPAD(IFNULL(MAX(SUBSTRING(id, 7, 4)), 0) + 1, 4, '0'))
        FROM material
        WHERE id LIKE CONCAT('WZ', DATE_FORMAT(NOW(), '%Y'), '%')
        """)
        new_id = cursor.fetchone()[0]
        return new_id or f"WZ{datetime.now().year}0001"
    except mysql.connector.Error as err:
        st.error(f"Error generating ID: {err}")
        return f"WZ{datetime.now().year}0001"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def generate_department_id():
    """Generate department ID using DEPT + 3-digit sequence"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT CONCAT('DEPT', LPAD(IFNULL(MAX(SUBSTRING(id, 5, 3)), 0) + 1, 3, '0'))
        FROM department
        WHERE id LIKE 'DEPT%'
        """)
        new_id = cursor.fetchone()[0]
        return new_id or "DEPT001"
    except mysql.connector.Error as err:
        st.error(f"Error generating ID: {err}")
        return "DEPT001"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def generate_user_id():
    """Generate user ID using USER + 4-digit sequence"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT CONCAT('USER', LPAD(IFNULL(MAX(SUBSTRING(user_id, 5, 4)), 0) + 1, 4, '0'))
        FROM user
        WHERE user_id LIKE 'USER%'
        """)
        new_id = cursor.fetchone()[0]
        return new_id or "USER0001"
    except mysql.connector.Error as err:
        st.error(f"Error generating ID: {err}")
        return "USER0001"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def generate_po_number():
    """Generate PO number using PO + date + 4-digit sequence"""
    conn = get_db_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y%m%d')

    try:
        cursor.execute(f"""
        SELECT CONCAT('PO{today}', LPAD(IFNULL(MAX(SUBSTRING(po_number, 11, 4)), 0) + 1, 4, '0'))
        FROM purchase_order
        WHERE po_number LIKE 'PO{today}%'
        """)
        new_id = cursor.fetchone()[0]
        return new_id or f"PO{today}0001"
    except mysql.connector.Error as err:
        st.error(f"Error generating ID: {err}")
        return f"PO{today}0001"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def generate_transaction_id():
    """Generate transaction ID using TR + date + 5-digit sequence"""
    conn = get_db_connection()
    cursor = conn.cursor()
    today = datetime.now().strftime('%Y%m%d')

    try:
        cursor.execute(f"""
        SELECT CONCAT('TR{today}', LPAD(IFNULL(MAX(SUBSTRING(transaction_id, 11, 5)), 0) + 1, 5, '0'))
        FROM inventory_transaction
        WHERE transaction_id LIKE 'TR{today}%'
        """)
        new_id = cursor.fetchone()[0]
        return new_id or f"TR{today}00001"
    except mysql.connector.Error as err:
        st.error(f"Error generating ID: {err}")
        return f"TR{today}00001"
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# =============================================
# Data Entry Functions (Updated with Auto-Generated IDs)
# =============================================

def add_material():
    st.subheader("Add New Material")

    with st.form("material_form"):
        # Auto-generate material ID
        material_id = generate_material_id()
        st.markdown(f"**Material ID:** `{material_id}`")

        name = st.text_input("Material Name*", max_chars=100)
        description = st.text_area("Description")
        category = st.selectbox("Category*", [
            "Electronics", "Mechanical", "Construction", 
            "Packaging", "Raw Materials", "Tools",
            "Safety Equipment", "Office Supplies"
        ])
        unit = st.selectbox("Unit*", ["pieces", "kg", "liters", "boxes", "units", "pallets"])
        current_quantity = st.number_input("Current Quantity*", min_value=0.0, step=0.1, value=0.0)
        min_quantity = st.number_input("Minimum Quantity*", min_value=0.0, step=0.1)
        max_quantity = st.number_input("Maximum Quantity*", min_value=0.0, step=0.1)
        location = st.text_input("Location (e.g., A1)", max_chars=50)

        submitted = st.form_submit_button("Add Material")

        if submitted:
            if not all([name, category, unit]):
                st.error("Please fill all required fields (marked with *)")
            else:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    cursor.execute("""
                    INSERT INTO material (id, name, description, category, unit, 
                                        current_quantity, min_quantity, max_quantity, location)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (material_id, name, description, category, unit, 
                          current_quantity, min_quantity, max_quantity, location))

                    conn.commit()
                    st.success(f"Material {name} added successfully with ID: {material_id}")
                except mysql.connector.Error as err:
                    st.error(f"Error: {err}")
                finally:
                    if conn.is_connected():
                        cursor.close()
                        conn.close()

def add_department():
    st.subheader("Add New Department")

    with st.form("department_form"):
        # Auto-generate department ID
        dept_id = generate_department_id()
        st.markdown(f"**Department ID:** `{dept_id}`")

        name = st.text_input("Department Name*", max_chars=100)
        manager = st.text_input("Manager Name", max_chars=100)
        contact = st.text_input("Contact Information", max_chars=50)

        submitted = st.form_submit_button("Add Department")

        if submitted:
            if not name:
                st.error("Department name is required")
            else:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    cursor.execute("""
                    INSERT INTO department (id, name, manager, contact)
                    VALUES (%s, %s, %s, %s)
                    """, (dept_id, name, manager, contact))

                    conn.commit()
                    st.success(f"Department {name} added successfully with ID: {dept_id}")
                except mysql.connector.Error as err:
                    st.error(f"Error: {err}")
                finally:
                    if conn.is_connected():
                        cursor.close()
                        conn.close()

def add_user():
    st.subheader("Add New User")

    # Get departments for dropdown
    departments = pd.read_sql("SELECT id, name FROM department", engine)
    dept_options = {row['id']: row['name'] for _, row in departments.iterrows()}

    with st.form("user_form"):
        # Auto-generate user ID
        user_id = generate_user_id()
        st.markdown(f"**User ID:** `{user_id}`")

        username = st.text_input("Username*", max_chars=50)
        password = st.text_input("Password*", type="password", max_chars=100)
        name = st.text_input("Full Name*", max_chars=100)
        role = st.selectbox("Role*", [
            "Manager", "Supervisor", "Operator", 
            "Clerk", "Technician", "Analyst"
        ])
        department_id = st.selectbox("Department", options=list(dept_options.keys()), 
                                    format_func=lambda x: dept_options[x])

        submitted = st.form_submit_button("Add User")

        if submitted:
            if not all([username, password, name, role]):
                st.error("Please fill all required fields (marked with *)")
            else:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    cursor.execute("""
                    INSERT INTO user (user_id, username, password, name, role, department_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """, (user_id, username, password, name, role, department_id))

                    conn.commit()
                    st.success(f"User {name} added successfully with ID: {user_id}")
                except mysql.connector.Error as err:
                    st.error(f"Error: {err}")
                finally:
                    if conn.is_connected():
                        cursor.close()
                        conn.close()

def add_purchase_order():
    st.subheader("Add New Purchase Order")

    # Get materials and departments for dropdowns
    materials = pd.read_sql("SELECT id, name FROM material", engine)
    mat_options = {row['id']: row['name'] for _, row in materials.iterrows()}

    departments = pd.read_sql("SELECT id, name FROM department", engine)
    dept_options = {row['id']: row['name'] for _, row in departments.iterrows()}

    with st.form("po_form"):
        # Auto-generate PO number
        po_number = generate_po_number()
        st.markdown(f"**PO Number:** `{po_number}`")

        material_id = st.selectbox("Material*", options=list(mat_options.keys()), 
                                 format_func=lambda x: mat_options[x])
        quantity = st.number_input("Quantity*", min_value=0.01, step=0.01)
        order_date = st.date_input("Order Date*", value=datetime.now())
        expected_delivery = st.date_input("Expected Delivery Date*", value=datetime.now())
        status = st.selectbox("Status*", [
            "Draft", "Submitted", "Approved", "Received", "Cancelled"
        ])
        department_id = st.selectbox("Requesting Department*", 
                                   options=list(dept_options.keys()), 
                                   format_func=lambda x: dept_options[x])

        submitted = st.form_submit_button("Add Purchase Order")

        if submitted:
            if expected_delivery < order_date:
                st.error("Delivery date cannot be before order date")
            else:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    cursor.execute("""
                    INSERT INTO purchase_order (po_number, material_id, quantity, 
                                              order_date, expected_delivery, status, department_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (po_number, material_id, quantity, 
                          order_date, expected_delivery, status, department_id))

                    conn.commit()
                    st.success(f"Purchase Order {po_number} added successfully!")
                except mysql.connector.Error as err:
                    st.error(f"Error: {err}")
                finally:
                    if conn.is_connected():
                        cursor.close()
                        conn.close()

def add_inventory_transaction():
    st.subheader("Add Inventory Transaction")

    # Get materials and users for dropdowns
    materials = pd.read_sql("SELECT id, name FROM material", engine)
    mat_options = {row['id']: row['name'] for _, row in materials.iterrows()}

    users = pd.read_sql("SELECT user_id, name FROM user", engine)
    user_options = {row['user_id']: row['name'] for _, row in users.iterrows()}

    # Get POs for reference (for IN transactions)
    purchase_orders = pd.read_sql("SELECT po_number FROM purchase_order", engine)
    po_options = purchase_orders['po_number'].tolist()

    with st.form("transaction_form"):
        # Auto-generate transaction ID
        transaction_id = generate_transaction_id()
        st.markdown(f"**Transaction ID:** `{transaction_id}`")

        material_id = st.selectbox("Material*", options=list(mat_options.keys()), 
                                 format_func=lambda x: mat_options[x])
        transaction_type = st.selectbox("Transaction Type*", 
                                      ["IN", "OUT", "TRANSFER", "SCRAP"])
        quantity = st.number_input("Quantity*", min_value=0.01, step=0.01)
        transaction_date = st.date_input("Transaction Date*", value=datetime.now())
        reference_number = st.selectbox("Reference PO (required for IN transactions)", 
                                      [""] + po_options) if transaction_type == "IN" else None
        user_id = st.selectbox("Performed By*", options=list(user_options.keys()), 
                             format_func=lambda x: user_options[x])
        notes = st.text_area("Notes")

        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            if transaction_type == "IN" and not reference_number:
                st.error("Reference PO is required for IN transactions")
            else:
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    # Adjust quantity for OUT/SCRAP transactions
                    actual_quantity = -abs(quantity) if transaction_type in ["OUT", "SCRAP"] else quantity

                    cursor.execute("""
                    INSERT INTO inventory_transaction (
                        transaction_id, material_id, transaction_type, 
                        quantity, transaction_date, reference_number, 
                        user_id, notes
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        transaction_id, material_id, transaction_type,
                        actual_quantity, transaction_date, reference_number if transaction_type == "IN" else None,
                        user_id, notes
                    ))

                    # Update material quantity
                    if transaction_type == "IN":
                        cursor.execute("""
                        UPDATE material 
                        SET current_quantity = current_quantity + %s
                        WHERE id = %s
                        """, (quantity, material_id))
                    elif transaction_type in ["OUT", "SCRAP"]:
                        cursor.execute("""
                        UPDATE material 
                        SET current_quantity = current_quantity - %s
                        WHERE id = %s
                        """, (quantity, material_id))

                    conn.commit()
                    st.success(f"Transaction {transaction_id} added successfully!")
                except mysql.connector.Error as err:
                    st.error(f"Error: {err}")
                finally:
                    if conn.is_connected():
                        cursor.close()
                        conn.close()

# =============================================
# Rest of your application (unchanged)
# =============================================

# Set page config and navigation (same as before)


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Dashboard", 
    "Inventory Analysis", 
    "Material Clustering",
    "Transaction Explorer",
    "Data Entry"
])

# Page routing (same structure as before)
if page == "Data Entry":
    st.title("📝 Data Entry")
    entry_type = st.selectbox("Select Data Type to Add", [
        "Material",
        "Department",
        "User",
        "Purchase Order",
        "Inventory Transaction"
    ])

    if entry_type == "Material":
        add_material()
    elif entry_type == "Department":
        add_department()
    elif entry_type == "User":
        add_user()
    elif entry_type == "Purchase Order":
        add_purchase_order()
    elif entry_type == "Inventory Transaction":
        add_inventory_transaction()

# =============================================
# Rest of your existing pages (unchanged)
# =============================================

elif page == "Dashboard":
    st.title("📊 Warehouse Management Dashboard")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_items = pd.read_sql("SELECT COUNT(*) FROM material", engine).iloc[0,0]
        st.metric("Total Materials", total_items)

    with col2:
        total_transactions = pd.read_sql("SELECT COUNT(*) FROM inventory_transaction", engine).iloc[0,0]
        st.metric("Total Transactions", total_transactions)

    with col3:
        low_stock = pd.read_sql(
            "SELECT COUNT(*) FROM material WHERE current_quantity < min_quantity", 
            engine
        ).iloc[0,0]
        st.metric("Low Stock Items", low_stock, delta_color="inverse")

    # Recent transactions
    st.subheader("Recent Inventory Movements")
    recent_transactions = pd.read_sql("""
    SELECT 
        t.transaction_id,
        m.name as material,
        t.transaction_type,
        t.quantity,
        t.transaction_date,
        d.name as department
    FROM inventory_transaction t
    JOIN material m ON t.material_id = m.id
    LEFT JOIN purchase_order po ON t.reference_number = po.po_number
    LEFT JOIN department d ON po.department_id = d.id
    ORDER BY t.transaction_date DESC
    LIMIT 10
    """, engine)
    st.dataframe(recent_transactions)

    # Inventory status pie chart
    st.subheader("Inventory Status")
    inventory_status = pd.read_sql("""
    SELECT 
        CASE 
            WHEN current_quantity < min_quantity THEN 'Below Minimum'
            WHEN current_quantity > max_quantity THEN 'Above Maximum'
            ELSE 'Within Range'
        END as status,
        COUNT(*) as count
    FROM material
    GROUP BY status
    """, engine)

    fig, ax = plt.subplots()
    ax.pie(
        inventory_status['count'],
        labels=inventory_status['status'],
        autopct='%1.1f%%',
        colors=['#ff9999','#66b3ff','#99ff99']
    )
    ax.axis('equal')
    st.pyplot(fig)


elif page == "Inventory Analysis":
    st.title("📈 Inventory Analysis")
    # Material selection
    materials = pd.read_sql("SELECT id, name FROM material", engine)
    selected_material = st.selectbox(
        "Select Material", 
        materials['name'],
        index=0
    )
    material_id = materials[materials['name'] == selected_material].iloc[0,0]

    # Get inventory history
    inventory_history = pd.read_sql(f"""
    SELECT 
        DATE(transaction_date) as date,
        SUM(quantity) as daily_change,
        SUM(SUM(quantity)) OVER (ORDER BY DATE(transaction_date)) as running_quantity
    FROM inventory_transaction
    WHERE material_id = '{material_id}'
    GROUP BY DATE(transaction_date)
    ORDER BY date
    """, engine)

    # Plot inventory trend
    st.subheader(f"Inventory Trend for {selected_material}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(inventory_history['date'], inventory_history['running_quantity'], label='Inventory Level')

    # Add min/max lines if available
    min_max = pd.read_sql(f"""
    SELECT min_quantity, max_quantity FROM material WHERE id = '{material_id}'
    """, engine).iloc[0]

    ax.axhline(y=min_max['min_quantity'], color='r', linestyle='--', label='Minimum Quantity')
    ax.axhline(y=min_max['max_quantity'], color='g', linestyle='--', label='Maximum Quantity')

    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Show transactions for this material
    st.subheader(f"Transactions for {selected_material}")
    material_transactions = pd.read_sql(f"""
    SELECT 
        transaction_id,
        transaction_type,
        quantity,
        transaction_date,
        reference_number,
        notes
    FROM inventory_transaction
    WHERE material_id = '{material_id}'
    ORDER BY transaction_date DESC
    LIMIT 20
    """, engine)
    st.dataframe(material_transactions)

elif page == "Material Clustering":
    st.title("🧩 Material Clustering Analysis")
    # Explanation
    st.markdown("""
    This section performs K-means clustering on materials based on:
    - Inventory characteristics (quantity, min, max)
    - Movement patterns (frequency, quantity)
    - Department usage
    """)

    if st.button("Run Clustering Analysis"):
        with st.spinner("Performing clustering..."):
            # Get data
            material_data = pd.read_sql("""
            WITH material_stats AS (
                SELECT 
                    m.id as material_id,
                    m.name as material_name,
                    m.category,
                    m.current_quantity,
                    m.min_quantity,
                    m.max_quantity,
                    COUNT(t.transaction_id) as transaction_count,
                    SUM(CASE WHEN t.transaction_type = 'IN' THEN t.quantity ELSE 0 END) as total_incoming,
                    SUM(CASE WHEN t.transaction_type IN ('OUT', 'SCRAP') THEN ABS(t.quantity) ELSE 0 END) as total_outgoing,
                    COUNT(DISTINCT po.department_id) as departments_used_in
                FROM material m
                LEFT JOIN inventory_transaction t ON m.id = t.material_id
                LEFT JOIN purchase_order po ON t.reference_number = po.po_number
                GROUP BY m.id, m.name, m.category, m.current_quantity, m.min_quantity, m.max_quantity
            )
            SELECT 
                *,
                current_quantity/max_quantity as inventory_ratio,
                total_incoming/NULLIF(transaction_count, 0) as avg_incoming,
                total_outgoing/NULLIF(transaction_count, 0) as avg_outgoing,
                CASE WHEN current_quantity < min_quantity THEN 1 ELSE 0 END as is_below_min
            FROM material_stats
            """, engine).fillna(0)

            # Preprocess
            features = material_data[[
                'current_quantity',
                'min_quantity',
                'max_quantity',
                'inventory_ratio',
                'transaction_count',
                'total_incoming',
                'total_outgoing',
                'avg_incoming',
                'avg_outgoing',
                'departments_used_in',
                'is_below_min'
            ]]

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Cluster
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_features)
            material_data['cluster'] = clusters

            # Show results
            st.success("Clustering completed!")

            # Cluster profiles
            st.subheader("Cluster Profiles")
            cluster_profiles = material_data.groupby('cluster').mean(numeric_only=True)
            st.dataframe(cluster_profiles)

            # Visualization
            st.subheader("Cluster Visualization (PCA Reduced)")

            # Reduce dimensions for visualization
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                reduced_features[:, 0], 
                reduced_features[:, 1], 
                c=material_data['cluster'], 
                cmap='viridis',
                alpha=0.7
            )

            ax.set_title("Material Clusters")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)

            # Show materials in each cluster
            st.subheader("Materials by Cluster")
            selected_cluster = st.selectbox(
                "Select Cluster to View Materials",
                sorted(material_data['cluster'].unique())
            )

            cluster_materials = material_data[material_data['cluster'] == selected_cluster][[
                'material_name', 'category', 'current_quantity', 
                'min_quantity', 'max_quantity', 'transaction_count'
            ]]
            st.dataframe(cluster_materials)


elif page == "Transaction Explorer":
    st.title("🔍 Transaction Explorer")
# Date range selector
    min_date, max_date = pd.read_sql("""
    SELECT MIN(transaction_date), MAX(transaction_date) FROM inventory_transaction
    """, engine).iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date)
    with col2:
        end_date = st.date_input("End Date", max_date)

    # Transaction type filter
    transaction_types = pd.read_sql("""
    SELECT DISTINCT transaction_type FROM inventory_transaction
    """, engine)['transaction_type'].tolist()
    selected_types = st.multiselect(
        "Transaction Types", 
        transaction_types,
        default=transaction_types
    )

    if st.button("Load Transactions"):
        query = f"""
        SELECT 
            t.transaction_id,
            m.name as material,
            t.transaction_type,
            t.quantity,
            t.transaction_date,
            d.name as department,
            t.notes
        FROM inventory_transaction t
        JOIN material m ON t.material_id = m.id
        LEFT JOIN purchase_order po ON t.reference_number = po.po_number
        LEFT JOIN department d ON po.department_id = d.id
        WHERE t.transaction_date BETWEEN '{start_date}' AND '{end_date}'
        AND t.transaction_type IN ({','.join([f"'{t}'" for t in selected_types])})
        ORDER BY t.transaction_date DESC
        """

        transactions = pd.read_sql(query, engine)
        st.dataframe(transactions)

        # Summary statistics
        st.subheader("Transaction Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Transactions", len(transactions))

        with col2:
            incoming = transactions[transactions['quantity'] > 0]['quantity'].sum()
            st.metric("Total Incoming", f"{incoming:.2f}")

        with col3:
            outgoing = transactions[transactions['quantity'] < 0]['quantity'].sum()
            st.metric("Total Outgoing", f"{outgoing:.2f}")

        # Time series plot
        st.subheader("Daily Transaction Volumes")
        daily_volumes = transactions.groupby(
            pd.to_datetime(transactions['transaction_date']).dt.date
        )['quantity'].agg(['count', 'sum']).reset_index()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(daily_volumes['transaction_date'], daily_volumes['count'])
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Transactions")
        ax.grid(True)
        st.pyplot(fig)

