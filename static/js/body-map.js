
document.addEventListener('DOMContentLoaded', function() {
    
    const bodyPartMappings = {
        'head': 6,               
        'chest': 15,             
        'abdomen': 16,           
        'left-arm': 7,           
        'right-arm': 7,          
        'left-leg': 10,          
        'right-leg': 10,         
        'skin': 17,              
        'general': 18            
    };

    
    const bodyImage = document.getElementById('bodyImage');
    const bodyPartLabel = document.getElementById('bodyPartLabel');
    let selectedBodyPart = null;
    
    
    function initializeBodyMap() {
        
        createClickableAreas();
        
        
        setupEventListeners();
    }
    
    
    function createClickableAreas() {
        const mapContainer = document.querySelector('.body-map-container');
        if (!mapContainer) return;
        
        
        createArea('head', 'Head', '50%', '12%', '20%', '15%');
        createArea('chest', 'Chest', '50%', '30%', '30%', '15%');
        createArea('abdomen', 'Abdomen', '50%', '45%', '30%', '12%');
        createArea('left-arm', 'Arms', '30%', '33%', '12%', '25%');
        createArea('right-arm', 'Arms', '70%', '33%', '12%', '25%');
        createArea('left-leg', 'Legs', '40%', '70%', '15%', '30%');
        createArea('right-leg', 'Legs', '60%', '70%', '15%', '30%');
        createArea('skin', 'Skin', '85%', '15%', '10%', '10%', true);
        createArea('general', 'General', '85%', '30%', '10%', '10%', true);
    }
    
    
    function createArea(id, name, left, top, width, height, isCircle = false) {
        const mapContainer = document.querySelector('.body-map-container');
        
        const area = document.createElement('div');
        area.id = `body-part-${id}`;
        area.className = 'body-part-area';
        area.dataset.name = name;
        area.dataset.id = bodyPartMappings[id];
        
        
        area.style.position = 'absolute';
        area.style.left = left;
        area.style.top = top;
        area.style.width = width;
        area.style.height = height;
        area.style.transform = 'translate(-50%, -50%)';
        area.style.cursor = 'pointer';
        area.style.zIndex = '10';
        
        
        area.style.border = '2px solid rgba(100, 116, 139, 0.2)';
        area.style.borderRadius = isCircle ? '50%' : '8px';
        area.style.backgroundColor = 'rgba(226, 232, 240, 0.2)';
        area.style.transition = 'all 0.3s ease';
        
        
        mapContainer.appendChild(area);
        
        
        area.addEventListener('click', handleBodyPartClick);
        area.addEventListener('mouseenter', handleBodyPartHover);
        area.addEventListener('mouseleave', handleBodyPartLeave);
        
        
        if (isCircle) {
            const label = document.createElement('div');
            label.textContent = name;
            label.style.position = 'absolute';
            label.style.top = '50%';
            label.style.left = '50%';
            label.style.transform = 'translate(-50%, -50%)';
            label.style.fontSize = '12px';
            label.style.color = '#64748b';
            label.style.pointerEvents = 'none';
            area.appendChild(label);
        }
    }
    
    
    function handleBodyPartClick() {
        
        const locationId = this.dataset.id;
        const locationName = this.dataset.name;
        
        if (locationId) {
            
            document.querySelectorAll('.body-part-area').forEach(part => {
                part.classList.remove('selected');
                part.style.backgroundColor = 'rgba(226, 232, 240, 0.2)';
                part.style.borderColor = 'rgba(100, 116, 139, 0.2)';
            });
            
            
            this.classList.add('selected');
            this.style.backgroundColor = 'rgba(186, 230, 253, 0.5)';
            this.style.borderColor = '#2e7bff';
            
            
            if (bodyPartLabel) {
                bodyPartLabel.textContent = locationName;
            }
            
            
            selectedBodyPart = locationId;
            
            
            const event = new CustomEvent('bodyPartSelected', {
                detail: {
                    locationId: locationId,
                    locationName: locationName
                }
            });
            document.dispatchEvent(event);
            
            
            const bodyAreaFilter = document.getElementById('bodyAreaFilter');
            if (bodyAreaFilter) {
                bodyAreaFilter.value = locationId;
                bodyAreaFilter.dispatchEvent(new Event('change'));
            }
        }
    }
    
    
    function handleBodyPartHover() {
        if (!this.classList.contains('selected')) {
            this.style.backgroundColor = 'rgba(209, 216, 224, 0.5)';
            
            
            if (bodyPartLabel) {
                bodyPartLabel.textContent = this.dataset.name;
            }
        }
    }
    
    
    function handleBodyPartLeave() {
        if (!this.classList.contains('selected')) {
            this.style.backgroundColor = 'rgba(226, 232, 240, 0.2)';
            
            
            if (bodyPartLabel) {
                const selectedPart = document.querySelector('.body-part-area.selected');
                if (selectedPart) {
                    bodyPartLabel.textContent = selectedPart.dataset.name;
                } else {
                    bodyPartLabel.textContent = 'Click on a body area';
                }
            }
        }
    }
    
    
    function setupEventListeners() {
        
        const bodyAreaFilter = document.getElementById('bodyAreaFilter');
        if (bodyAreaFilter) {
            bodyAreaFilter.addEventListener('change', function() {
                const locationId = this.value;
                if (locationId) {
                    
                    const bodyPart = document.querySelector(`.body-part-area[data-id="${locationId}"]`);
                    if (bodyPart) {
                        bodyPart.click();
                    }
                } else {
                    
                    document.querySelectorAll('.body-part-area').forEach(part => {
                        part.classList.remove('selected');
                        part.style.backgroundColor = 'rgba(226, 232, 240, 0.2)';
                        part.style.borderColor = 'rgba(100, 116, 139, 0.2)';
                    });
                    
                    
                    if (bodyPartLabel) {
                        bodyPartLabel.textContent = 'Click on a body area';
                    }
                    
                    selectedBodyPart = null;
                }
            });
        }
    }
    
    
    initializeBodyMap();
}); 