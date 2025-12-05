package com.wealthflow.backend.model;

import jakarta.persistence.*;

@Entity
@Table(name = "allocations")
public class Allocation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // which asset's allocation
    @ManyToOne(optional = false)
    @JoinColumn(name = "asset_id")
    private Asset asset;

    // recommendation that this allocation belongs to
    @ManyToOne
    @JoinColumn(name = "recommendation_id")
    private Recommendation recommendation;

    // recommended percentage for that asset
    @Column(nullable = false)
    private Double percentage;

    public Allocation() {
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Asset getAsset() {
        return asset;
    }

    public void setAsset(Asset asset) {
        this.asset = asset;
    }

    public Recommendation getRecommendation() {
        return recommendation;
    }

    public void setRecommendation(Recommendation recommendation) {
        this.recommendation = recommendation;
    }

    public Double getPercentage() {
        return percentage;
    }

    public void setPercentage(Double percentage) {
        this.percentage = percentage;
    }
}
