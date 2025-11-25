package com.wealthflow.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "recommendations")
public class Recommendation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // recommendation belongs to one specific user
    @ManyToOne(optional = false)
    @JoinColumn(name = "user_profile_id")
    private UserProfile userProfile;

    // recommendation contains many allocations
    @OneToMany(mappedBy = "recommendation", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<Allocation> allocations;

    @Column(nullable = false)
    private LocalDateTime generatedAt;

    public Recommendation() {
    }

    public Recommendation(UserProfile userProfile, List<Allocation> allocations, LocalDateTime generatedAt) {
        this.userProfile = userProfile;
        this.allocations = allocations;
        this.generatedAt = generatedAt;
    }

    public Recommendation(Long id, UserProfile userProfile, List<Allocation> allocations, LocalDateTime generatedAt) {
        this.id = id;
        this.userProfile = userProfile;
        this.allocations = allocations;
        this.generatedAt = generatedAt;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public UserProfile getUserProfile() {
        return userProfile;
    }

    public void setUserProfile(UserProfile userProfile) {
        this.userProfile = userProfile;
    }

    public List<Allocation> getAllocations() {
        return allocations;
    }

    public void setAllocations(List<Allocation> allocations) {
        this.allocations = allocations;
    }

    public LocalDateTime getGeneratedAt() {
        return generatedAt;
    }

    public void setGeneratedAt(LocalDateTime generatedAt) {
        this.generatedAt = generatedAt;
    }
}
