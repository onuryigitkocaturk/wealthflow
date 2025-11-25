package com.wealthflow.backend.model;

import com.wealthflow.backend.model.enums.AssetType;
import com.wealthflow.backend.model.enums.RiskLevel;
import jakarta.persistence.*;

@Entity
@Table(name = "assets")
public class Asset {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true)
    private String symbol;

    @Column(nullable = false)
    private String name;

    @Enumerated(EnumType.STRING)
    private AssetType assetType;

    @Enumerated(EnumType.STRING)
    private RiskLevel riskLevel;

    @Column(nullable = true)
    private Double esgScore;

    public Asset() {
    }

    public Asset(String symbol, String name, AssetType assetType, RiskLevel riskLevel, Double esgScore) {
        this.symbol = symbol;
        this.name = name;
        this.assetType = assetType;
        this.riskLevel = riskLevel;
        this.esgScore = esgScore;
    }

    public Asset(Long id, String symbol, String name, AssetType assetType, RiskLevel riskLevel, Double esgScore) {
        this.id = id;
        this.symbol = symbol;
        this.name = name;
        this.assetType = assetType;
        this.riskLevel = riskLevel;
        this.esgScore = esgScore;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public AssetType getAssetType() {
        return assetType;
    }

    public void setAssetType(AssetType assetType) {
        this.assetType = assetType;
    }

    public RiskLevel getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(RiskLevel riskLevel) {
        this.riskLevel = riskLevel;
    }

    public Double getEsgScore() {
        return esgScore;
    }

    public void setEsgScore(Double esgScore) {
        this.esgScore = esgScore;
    }
}
