from flask import Flask, render_template
import pandas as pd
import os
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv('C:\\Users\\Tharun\\Downloads\\MCI_2014_to_2019.csv', sep=',')
    df['Total'] = 1
    df.head()
    
    major_crime_indicator = df['MCI'].value_counts().reset_index()
    major_crime_indicator.columns = ['MCI', 'crime_count']
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ct = major_crime_indicator.sort_values(by='crime_count', ascending=False)
    ax = ct.plot.bar(x='MCI', y='crime_count', ax=ax)
    ax.set_xlabel('Offence')
    ax.set_ylabel('Total Number of Criminal Cases from 2014 to 2019')
    ax.set_title('Major Crime Indicator', color='red', fontsize=25)
    fig.savefig('static/graph1.png')
    plt.close(fig)
    
    df2 = df[df['occurrenceyear'] > 2013]
    yearwise_total_crime = df2.groupby('occurrenceyear').size()
    
    fig2, ax2 = plt.subplots(figsize=(13, 6))
    ct = yearwise_total_crime.sort_values(ascending=True)
    ax2 = ct.plot.line(ax=ax2)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Number of Criminal Cases throughout 2014 to 2019')
    ax2.set_title('Yearwise Total Criminal Cases throughout 2014 to 2019', color='red', fontsize=25)
    ax2.grid(linestyle='-')
    fig2.savefig('static/graph2.png')
    plt.close(fig2)
    
    # Plotting Pie chart for crime according to premisetype
    premise_type = df.groupby('premisetype').size()
    premise_type.head()
    
    labels = ['Outside', 'Apartment', 'Commercial', 'House', 'Other']
    count = [54253, 49996, 41681, 37927, 23178]
    
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    explode = (0, 0, 0, 0, 0)
    ax3.pie(count, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
    ax3.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax3.set_title('Proportion of Crimes according to Premise Type', color='red', fontsize=25)
    fig3.savefig('static/graph3.png')
    plt.close(fig3)
    
    assault = df[df['MCI'] == 'Assault']
    assault_types = assault.groupby('offence').size()
    fig4, ax4 = plt.subplots(figsize=(15, 6))
    ct = assault_types.sort_values(ascending=False)
    ax4 = ct.plot.bar(ax=ax4)
    ax4.set_xlabel("Types of Assault")
    ax4.set_ylabel('Number of Occurrences')
    ax4.set_title('Assault Crimes in Toronto', color='red', fontsize=25)
    fig4.savefig('static/graph4.png')
    plt.close(fig4)
    
    # Plotting line graph for crime types by hour of day
    hour_crime_group = df.groupby(['occurrencehour', 'MCI'], as_index=False).agg({'Total': 'sum'})
    fig5, ax5 = plt.subplots(figsize=(15, 10))
    fig, ax5 = plt.subplots(figsize=(15, 10))
    hour_crime_group.groupby("MCI").plot(x="occurrencehour", y="Total", ax=ax5, linewidth=5)
    ax5.set_xlabel('Hour (24-hour clock)')
    ax5.set_ylabel('Number of Occurrences')
    ax5.set_title('Crime Types by Hour of Day in Toronto', color='red', fontsize=25)
    ax5.grid(linestyle='')
    leg = plt.legend([v[0] for v in hour_crime_group.groupby('MCI')['MCI']])
    leg_lines = leg.get_lines()
    leg_texts = leg.get_texts()
    # Bulk-set the properties of all Lines and texts
    plt.setp(leg_lines, linewidth=4)
    plt.setp(leg_texts, fontsize='x-large')
    fig5.savefig('static/graph5.png')
    plt.close(fig5)
    
    location_group = df.groupby('Neighbourhood').size().sort_values(ascending=False).head(20)
    fig6, ax6 = plt.subplots(figsize=(15, 6))
    ax6 = location_group.sort_values(ascending=False).plot.bar()
    ax6.set_xlabel('Neighbourhoods')
    ax6.set_ylabel('Number of Occurrences')
    ax6.set_title('Neighbourhoods with Most Crimes', color='red', fontsize=25)
    fig6.savefig('static/graph6.png')
    plt.close(fig6)
    
    # Plotting Heatmap for Major Crime Indicator by Month
    mci_monthwise = df.groupby(['occurrencemonth', 'MCI'], as_index=False).agg({"Total": "sum"})
    fig7 = plt.figure(figsize=(15, 7))
    crime_count = mci_monthwise.pivot("MCI", "occurrencemonth", "Total")
    plt.yticks(rotation=1)
    ax7 = sns.heatmap(crime_count, cmap="YlGnBu", linewidths=.5)
    plt.title("Major Crime Indicators by Month", color="red", fontsize=14)
    fig7.savefig('static/graph7.png')
    plt.close(fig7)
    
    np = 'C:\\Users\\Tharun\\Downloads\\India Shapefile With Kashmir\\India Shape\\india_ds.shp'
    regions = gpd.read_file(np)
    print(regions.head())
    regions.columns = regions.columns.str.strip()
    
    crime_by_neighbourhood = df.groupby('Neighbourhood')['Total'].sum().reset_index()
    merged = regions.set_index('STATE').join(crime_by_neighbourhood.set_index('Neighbourhood'))
    fig8, ax8 = plt.subplots(1, figsize=(40, 20))
    print(merged.head())
    ax8.axis('off')
    ax8.set_title('Neighbourhoods with Most Crimes in Toronto', fontsize=40, color='red')
    merged.plot(column='DST_ID', cmap='Oranges_r', linewidth=0.8, ax=ax8, edgecolor='0.8', legend=True)
    fig8.savefig('static/graph8.png')
    plt.close(fig8)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
